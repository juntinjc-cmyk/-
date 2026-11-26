"""
即時人物偵測：用 dxcam 擷取螢幕畫面，用 YOLO 偵測人物並畫框。

用法範例：
    python person_detector.py                      # 擷取整個螢幕，目標 60 FPS
    python person_detector.py --size 640           # 只擷取螢幕中央 640x640（較快）
    python person_detector.py --region 0 0 1280 720
    python person_detector.py --no-show            # 不開預覽視窗，只在終端機印 FPS

按 q 或 Esc 結束。
"""

import argparse
import ctypes
import time

import cv2
import dxcam
import numpy as np
import torch
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO 資料集裡 "person" 的類別編號
WINDOW_NAME = "Person Detector"


def build_parser():
    p = argparse.ArgumentParser(description="dxcam + YOLO 即時人物偵測")
    p.add_argument("--model", default="yolo11n.pt",
                   help="YOLO 模型檔（第一次執行會自動下載）。n 最快，s/m 較準但較慢")
    p.add_argument("--fps", type=int, default=60, help="擷取的目標 FPS")
    p.add_argument("--size", type=int, default=0,
                   help="擷取螢幕中央 size x size 的區域；0（預設）代表整個螢幕")
    p.add_argument("--region", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                   help="自訂擷取區域（會覆蓋 --size）")
    p.add_argument("--imgsz", type=int, nargs="+", default=[640],
                   help="模型輸入尺寸：一個數字（正方形，例如 640）或「高 寬」兩個數字"
                        "（例如 736 1280，適合 16:9 螢幕）。越大越能抓到遠處小人物，但越慢。"
                        "用 .engine 時必須跟轉換時的 imgsz 一樣")
    p.add_argument("--conf", type=float, default=0.4, help="信心門檻，低於此值的框不顯示")
    p.add_argument("--monitor", type=int, default=0, help="要擷取的螢幕編號")
    p.add_argument("--device", default=None, help="例如 0（第一張 GPU）或 cpu；預設自動選擇")
    p.add_argument("--show-scale", type=float, default=0.5, help="預覽視窗縮放比例")
    p.add_argument("--no-show", action="store_true", help="不顯示預覽視窗")
    p.add_argument("--cpu-preprocess", action="store_true",
                   help="改回 ultralytics 內建的 CPU 前處理（GPU 前處理有問題時用）")
    return p


def compute_region(args, screen_w, screen_h):
    if args.region:
        return tuple(args.region)
    if args.size <= 0:
        return (0, 0, screen_w, screen_h)
    s = min(args.size, screen_w, screen_h)
    left = (screen_w - s) // 2
    top = (screen_h - s) // 2
    return (left, top, left + s, top + s)


def exclude_hwnd_from_capture(hwnd):
    """讓視窗不會被螢幕擷取到，避免它出現在偵測的畫面裡（畫面套畫面）。

    使用 Windows 的 SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)，
    需要 Windows 10 2004（20H1）以上。視窗在你眼中照常顯示，只是擷取畫面裡看不到它。
    """
    WDA_EXCLUDEFROMCAPTURE = 0x00000011
    if not ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
        print("[警告] 排除擷取失敗（Windows 版本可能太舊），視窗可能會出現在擷取畫面中")
        return False
    return True


def exclude_window_from_capture(title):
    hwnd = ctypes.windll.user32.FindWindowW(None, title)
    if not hwnd:
        print("[警告] 找不到預覽視窗，無法排除擷取")
        return False
    return exclude_hwnd_from_capture(hwnd)


def draw_boxes(frame, boxes, confs):
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = "person"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(y1, th + 4)
        cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 4, y_text), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y_text - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


class GpuPreprocessor:
    """把螢幕畫面轉成模型輸入，重活交給 GPU。

    ultralytics 內建的前處理全在 CPU 上用 numpy 做（4K 畫面縮小、BGR 轉 RGB、
    轉置、轉浮點數），而且很吃 Python 的 GIL，跟疊加視窗搶 CPU 時會明顯變慢。
    這裡只在 CPU 上用 OpenCV 縮小一次（多執行緒、很快），其餘都在 GPU 上做。
    """

    def __init__(self, target_hw, device):
        self.th, self.tw = target_hw
        self.device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
        self.pinned = None

    def __call__(self, frame):
        h, w = frame.shape[:2]
        r = min(self.th / h, self.tw / w)
        nw, nh = min(round(w * r), self.tw), min(round(h * r), self.th)
        small = frame if (nw, nh) == (w, h) else cv2.resize(
            frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # 用 pinned memory 讓 CPU -> GPU 傳輸更快
        if self.pinned is None or tuple(self.pinned.shape) != small.shape:
            self.pinned = torch.empty(small.shape, dtype=torch.uint8).pin_memory()
        self.pinned.numpy()[...] = small
        g = self.pinned.to(self.device, non_blocking=True)

        x = g.permute(2, 0, 1).flip(0).float().div_(255.0)  # HWC BGR -> CHW RGB, 0~1
        canvas = torch.full((1, 3, self.th, self.tw), 114 / 255.0,
                            device=self.device, dtype=torch.float32)  # 灰色補邊，跟 ultralytics 一樣
        canvas[0, :, :nh, :nw] = x
        return canvas, r


class Detector:
    """載入模型並暖機；呼叫 detector(frame) 回傳 (框 xyxy 整數陣列, 信心分數陣列)，座標對應原始畫面。"""

    def __init__(self, args):
        use_cuda = torch.cuda.is_available()
        device = args.device if args.device is not None else (0 if use_cuda else "cpu")
        gpu = use_cuda and str(device) != "cpu"
        print(f"[資訊] 裝置: {'CUDA - ' + torch.cuda.get_device_name(0) if gpu else 'CPU'}")
        print(f"[資訊] PyTorch {torch.__version__}，CUDA {torch.version.cuda}")
        if not gpu:
            print("[警告] 沒有使用 GPU，60 FPS 基本上跑不到。請安裝 CUDA 版 PyTorch。")

        self.model = YOLO(args.model)

        if len(args.imgsz) == 1:
            imgsz, (h, w) = args.imgsz[0], (args.imgsz[0], args.imgsz[0])
        else:
            imgsz, (h, w) = list(args.imgsz[:2]), tuple(args.imgsz[:2])

        # 預測參數只算一次，迴圈裡重複使用
        self.kwargs = dict(classes=[PERSON_CLASS_ID], conf=args.conf, imgsz=imgsz,
                           device=device, verbose=False)
        if gpu and not args.model.endswith(".engine"):
            # .engine 轉換時已經是 FP16，不用再指定。
            # 新版 ultralytics 把 half 改名成 quantize，舊版沒有 quantize，兩種都相容
            from ultralytics.cfg import DEFAULT_CFG_DICT
            if "quantize" in DEFAULT_CFG_DICT:
                self.kwargs["quantize"] = 16
            else:
                self.kwargs["half"] = True

        self.pre = GpuPreprocessor((h, w), device) if gpu and not args.cpu_preprocess else None
        print(f"[資訊] 前處理: {'GPU' if self.pre else 'CPU（ultralytics 內建）'}")

        # 暖機：前幾次推論比較慢，先跑幾次空白圖
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(3):
            self(dummy)

    def __call__(self, frame):
        """self.speed 會記錄 ultralytics 內部的「前處理 / 推論 / 後處理」毫秒數。"""
        if self.pre is not None:
            x, r = self.pre(frame)
            res = self.model.predict(x, **self.kwargs)[0]
            xyxy = (res.boxes.xyxy / r).round().int().cpu().numpy()
        else:
            res = self.model.predict(frame, **self.kwargs)[0]
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
        self.speed = res.speed
        return xyxy, res.boxes.conf.cpu().numpy()


def main():
    args = build_parser().parse_args()
    detector = Detector(args)

    # ---- 螢幕擷取 ----
    camera = dxcam.create(output_idx=args.monitor, output_color="BGR")
    region = compute_region(args, camera.width, camera.height)
    print(f"[資訊] 螢幕 {camera.width}x{camera.height}，擷取區域 {region}，目標 {args.fps} FPS")
    # video_mode=True：dxcam 在背景執行緒以固定 FPS 擷取，畫面沒變也會提供最新一幀
    camera.start(region=region, target_fps=args.fps, video_mode=True)

    if not args.no_show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(WINDOW_NAME, 0, 0)
        cv2.waitKey(1)  # 讓視窗真正建立出來，才找得到它
        exclude_window_from_capture(WINDOW_NAME)

    fps_ema = 0.0
    last = time.perf_counter()
    last_print = last
    # 每秒統計各階段平均花多少毫秒，用來找出是哪裡拖慢 FPS
    t_cap = t_inf = t_show = 0.0
    sp = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    n = 0

    try:
        while True:
            t0 = time.perf_counter()
            frame = camera.get_latest_frame()
            if frame is None:
                continue
            t1 = time.perf_counter()

            boxes, confs = detector(frame)
            t2 = time.perf_counter()

            if not args.no_show:
                view = np.ascontiguousarray(frame)  # 確保 OpenCV 可以直接在上面畫
                draw_boxes(view, boxes, confs)
                cv2.putText(view, f"FPS {fps_ema:5.1f} | persons {len(boxes)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                if args.show_scale != 1.0:
                    view = cv2.resize(view, None, fx=args.show_scale, fy=args.show_scale,
                                      interpolation=cv2.INTER_LINEAR)
                cv2.imshow(WINDOW_NAME, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            t3 = time.perf_counter()

            t_cap += t1 - t0
            t_inf += t2 - t1
            for k in sp:
                sp[k] += detector.speed.get(k) or 0.0
            t_show += t3 - t2
            n += 1

            dt = t3 - last
            last = t3
            fps_ema = (1.0 / dt) if fps_ema == 0 else fps_ema * 0.9 + (1.0 / dt) * 0.1

            if t3 - last_print >= 1.0:
                print(f"FPS {fps_ema:5.1f} | 擷取等待 {t_cap / n * 1000:5.1f} ms"
                      f" | 偵測 {t_inf / n * 1000:5.1f} ms"
                      f"（前處理 {sp['preprocess'] / n:4.1f} / 推論 {sp['inference'] / n:4.1f}"
                      f" / 後處理 {sp['postprocess'] / n:4.1f}）"
                      f" | 畫框顯示 {t_show / n * 1000:5.1f} ms | 人數 {len(boxes)}")
                t_cap = t_inf = t_show = 0.0
                sp = dict.fromkeys(sp, 0.0)
                n = 0
                last_print = t3
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        del camera
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

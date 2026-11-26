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


def parse_args():
    p = argparse.ArgumentParser(description="dxcam + YOLO 即時人物偵測")
    p.add_argument("--model", default="yolo11n.pt",
                   help="YOLO 模型檔（第一次執行會自動下載）。n 最快，s/m 較準但較慢")
    p.add_argument("--fps", type=int, default=60, help="擷取的目標 FPS")
    p.add_argument("--size", type=int, default=0,
                   help="擷取螢幕中央 size x size 的區域；0（預設）代表整個螢幕")
    p.add_argument("--region", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                   help="自訂擷取區域（會覆蓋 --size）")
    p.add_argument("--imgsz", type=int, default=640,
                   help="模型輸入尺寸。越小越快；全螢幕時遠處小人物抓不到可調大到 960 或 1280")
    p.add_argument("--conf", type=float, default=0.4, help="信心門檻，低於此值的框不顯示")
    p.add_argument("--monitor", type=int, default=0, help="要擷取的螢幕編號")
    p.add_argument("--device", default=None, help="例如 0（第一張 GPU）或 cpu；預設自動選擇")
    p.add_argument("--show-scale", type=float, default=0.5, help="預覽視窗縮放比例")
    p.add_argument("--no-show", action="store_true", help="不顯示預覽視窗")
    return p.parse_args()


def compute_region(args, screen_w, screen_h):
    if args.region:
        return tuple(args.region)
    if args.size <= 0:
        return (0, 0, screen_w, screen_h)
    s = min(args.size, screen_w, screen_h)
    left = (screen_w - s) // 2
    top = (screen_h - s) // 2
    return (left, top, left + s, top + s)


def exclude_window_from_capture(title):
    """讓預覽視窗不會被螢幕擷取到，避免全螢幕時出現「畫面套畫面」。

    使用 Windows 的 SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)，
    需要 Windows 10 2004（20H1）以上。視窗在你眼中照常顯示，只是擷取畫面裡看不到它。
    """
    WDA_EXCLUDEFROMCAPTURE = 0x00000011
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        print("[警告] 找不到預覽視窗，無法排除擷取")
        return False
    if not user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
        print("[警告] 排除擷取失敗（Windows 版本可能太舊），預覽視窗可能會出現在畫面中")
        return False
    return True


def draw_boxes(frame, boxes, confs):
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(y1, th + 4)
        cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 4, y_text), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y_text - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    args = parse_args()

    # ---- 模型 ----
    use_cuda = torch.cuda.is_available()
    device = args.device if args.device is not None else (0 if use_cuda else "cpu")
    half = use_cuda and str(device) != "cpu"  # GPU 上用 FP16 加速
    print(f"[資訊] 裝置: {'CUDA - ' + torch.cuda.get_device_name(0) if half else 'CPU'}")
    print(f"[資訊] PyTorch {torch.__version__}，CUDA {torch.version.cuda}")
    if not half:
        print("[警告] 沒有使用 GPU，60 FPS 基本上跑不到。請安裝 CUDA 版 PyTorch。")

    model = YOLO(args.model)
    # 暖機：第一次推論比較慢，先跑一次空白圖
    model.predict(np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8),
                  imgsz=args.imgsz, device=device, half=half, verbose=False)

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
    n = 0

    try:
        while True:
            t0 = time.perf_counter()
            frame = camera.get_latest_frame()
            if frame is None:
                continue
            t1 = time.perf_counter()

            result = model.predict(frame, classes=[PERSON_CLASS_ID], conf=args.conf,
                                   imgsz=args.imgsz, device=device, half=half,
                                   verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
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
            t_show += t3 - t2
            n += 1

            dt = t3 - last
            last = t3
            fps_ema = (1.0 / dt) if fps_ema == 0 else fps_ema * 0.9 + (1.0 / dt) * 0.1

            if t3 - last_print >= 1.0:
                print(f"FPS {fps_ema:5.1f} | 擷取等待 {t_cap / n * 1000:5.1f} ms"
                      f" | 偵測 {t_inf / n * 1000:5.1f} ms"
                      f" | 畫框顯示 {t_show / n * 1000:5.1f} ms | 人數 {len(boxes)}")
                t_cap = t_inf = t_show = 0.0
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

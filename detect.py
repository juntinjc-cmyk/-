"""
即時辨識遊戲畫面中的人。

用法：
  python detect.py                       # 擷取整個主螢幕
  python detect.py --window "遊戲視窗標題"   # 只擷取指定視窗
  python detect.py --source test.png     # 辨識圖片或影片檔
  python detect.py --save-frames         # 每隔一段時間存下畫面，之後可拿來訓練
  python detect.py --model yolo11s.pt --imgsz 640   # 追求高 FPS

按 q 離開。
"""
import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def get_window_region(title):
    """依視窗標題找出視窗在螢幕上的位置（只支援 Windows）。"""
    import pygetwindow as gw

    wins = [w for w in gw.getWindowsWithTitle(title) if w.width > 0 and w.height > 0]
    if not wins:
        raise SystemExit(f"找不到標題包含「{title}」的視窗")
    w = wins[0]
    return {"left": w.left, "top": w.top, "width": w.width, "height": w.height}


def draw(frame, result, fps=None):
    boxes = result.boxes
    for (x1, y1, x2, y2), conf in zip(boxes.xyxy.cpu().numpy().astype(int), boxes.conf.cpu().numpy()):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    text = f"People: {len(boxes)}"
    if fps:
        text += f"   FPS: {fps:.0f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


def run_file(model, args):
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    for r in model.predict(args.source, classes=[0], conf=args.conf, device=args.device,
                           imgsz=args.imgsz, stream=True, verbose=False):
        frame = draw(r.orig_img.copy(), r)
        print(f"{Path(r.path).name}: {len(r.boxes)} 人")
        cv2.imshow("detect", frame)
        if cv2.waitKey(0 if args.source.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")) else 1) & 0xFF == ord("q"):
            break
        cv2.imwrite(str(out_dir / Path(r.path).name), frame)
    cv2.destroyAllWindows()


class Grabber:
    """在背景執行緒持續擷取畫面，辨識時直接拿最新的一張，不用等擷取。

    優先用 dxcam（Windows 桌面複製 API，很快），沒裝就退回 mss。
    """

    def __init__(self, window=None):
        self.window = window
        self.frame = None
        self.capture_ms = 0.0
        self.running = True
        self.backend = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
        if not self._ready.wait(5):
            raise SystemExit("擷取畫面失敗")

    def _region(self, mon_w, mon_h, mon_left=0, mon_top=0):
        if not self.window:
            return None
        r = get_window_region(self.window)
        l = max(r["left"] - mon_left, 0)
        t = max(r["top"] - mon_top, 0)
        return (l, t, min(l + r["width"], mon_w), min(t + r["height"], mon_h))

    def _loop(self):
        try:
            import dxcam

            cam = dxcam.create(output_color="BGR")
            region = self._region(cam.width, cam.height)
            self.backend = "dxcam"
            while self.running:
                t0 = time.perf_counter()
                f = cam.grab(region=region)
                if f is None:  # 畫面沒變化時 dxcam 會回傳 None
                    time.sleep(0.001)
                    continue
                self._store(f, t0)
        except ImportError:
            import mss

            with mss.mss() as sct:
                mon = sct.monitors[1]
                if self.window:
                    r = get_window_region(self.window)
                else:
                    r = mon
                self.backend = "mss"
                while self.running:
                    t0 = time.perf_counter()
                    f = cv2.cvtColor(np.array(sct.grab(r)), cv2.COLOR_BGRA2BGR)
                    self._store(f, t0)

    def _store(self, f, t0):
        with self._lock:
            self.frame = f
            self.capture_ms = (time.perf_counter() - t0) * 1000
        self._ready.set()

    def latest(self):
        with self._lock:
            return self.frame

    def stop(self):
        self.running = False


def run_screen(model, args):
    save_dir = Path("captured_frames")
    if args.save_frames:
        save_dir.mkdir(exist_ok=True)
    last_save = 0.0

    grabber = Grabber(args.window)
    print(f"擷取方式：{grabber.backend}")
    if grabber.backend == "mss":
        print("提示：pip install dxcam 可以讓擷取快很多")

    half = args.device != "cpu"
    fps = 0.0
    prev = time.perf_counter()
    try:
        while True:
            frame = grabber.latest().copy()

            if args.save_frames and time.time() - last_save >= args.save_every:
                cv2.imwrite(str(save_dir / f"{int(time.time() * 1000)}.png"), frame)
                last_save = time.time()

            t0 = time.perf_counter()
            r = model.predict(frame, classes=[0], conf=args.conf, device=args.device,
                              imgsz=args.imgsz, half=half, verbose=False)[0]
            infer_ms = (time.perf_counter() - t0) * 1000

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1 / max(now - prev, 1e-6))  # 平滑一點比較好讀
            prev = now

            if args.no_preview:
                if int(now) != int(now - 1 / max(fps, 1)):
                    print(f"FPS {fps:5.1f} | 人數 {len(r.boxes)} | 辨識 {infer_ms:.1f} ms "
                          f"| 擷取 {grabber.capture_ms:.1f} ms")
                continue

            show = draw(frame, r, fps)
            cv2.putText(show, f"infer {infer_ms:.1f}ms  grab {grabber.capture_ms:.1f}ms",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            if args.scale != 1:
                show = cv2.resize(show, None, fx=args.scale, fy=args.scale,
                                  interpolation=cv2.INTER_NEAREST)
            cv2.imshow("detect (q to quit)", show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        grabber.stop()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11m.pt", help="模型檔，之後換成自己訓練的 best.pt")
    p.add_argument("--source", help="圖片 / 影片 / 資料夾；不給就擷取螢幕")
    p.add_argument("--window", help="只擷取標題包含這段文字的視窗")
    p.add_argument("--conf", type=float, default=0.25, help="信心門檻，越高越嚴格")
    p.add_argument("--imgsz", type=int, default=960, help="辨識解析度，人很小時調大")
    p.add_argument("--device", default="0", help="0 = 第一張顯卡，cpu = 不用顯卡")
    p.add_argument("--scale", type=float, default=0.6, help="預覽視窗縮放")
    p.add_argument("--no-preview", action="store_true", help="不顯示預覽視窗，只印出結果（最快）")
    p.add_argument("--save-frames", action="store_true", help="定期存下畫面供之後訓練")
    p.add_argument("--save-every", type=float, default=1.0, help="存畫面的間隔（秒）")
    args = p.parse_args()

    model = YOLO(args.model)
    if args.source:
        run_file(model, args)
    else:
        run_screen(model, args)


if __name__ == "__main__":
    main()

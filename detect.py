"""
即時辨識遊戲畫面中的人。

用法：
  python detect.py                       # 擷取整個主螢幕
  python detect.py --window "遊戲視窗標題"   # 只擷取指定視窗
  python detect.py --source test.png     # 辨識圖片或影片檔
  python detect.py --save-frames         # 每隔一段時間存下畫面，之後可拿來訓練

按 q 離開。
"""
import argparse
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


def run_screen(model, args):
    import mss

    save_dir = Path("captured_frames")
    if args.save_frames:
        save_dir.mkdir(exist_ok=True)
    last_save = 0.0
    prev = time.time()

    with mss.mss() as sct:
        region = get_window_region(args.window) if args.window else sct.monitors[1]
        while True:
            frame = cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)

            if args.save_frames and time.time() - last_save >= args.save_every:
                cv2.imwrite(str(save_dir / f"{int(time.time() * 1000)}.png"), frame)
                last_save = time.time()

            r = model.predict(frame, classes=[0], conf=args.conf, device=args.device,
                              imgsz=args.imgsz, verbose=False)[0]
            now = time.time()
            fps, prev = 1 / max(now - prev, 1e-6), now

            show = draw(frame, r, fps)
            if args.scale != 1:
                show = cv2.resize(show, None, fx=args.scale, fy=args.scale)
            cv2.imshow("detect (q to quit)", show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
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

"""
收集訓練用的畫面（真實影片：演唱會、樂團、車聚、街拍、運動等）。

兩種方式：
1. 從影片檔抽畫面（最推薦，畫質最好）：
       python training/collect_frames.py --video 我的影片.mp4 另一支.mp4 --every 2
2. 播放影片時，每隔幾秒截一次螢幕：
       python training/collect_frames.py --screen --every 3 --minutes 60

畫面存到 dataset/raw/。很像的畫面（例如鏡頭沒動）會自動跳過，避免一堆重複的圖。
按 Ctrl + C 可以隨時停止。
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SAVE_WIDTH = 1920  # 存成 1920 寬就夠了（訓練用 960），檔案也比較小


def dhash(img, size=16):
    """畫面的「指紋」：用來判斷兩張圖是不是幾乎一樣。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


class Saver:
    def __init__(self, out_dir, min_diff):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.min_diff = min_diff
        self.last_hash = None
        self.saved = 0
        self.skipped = 0

    def save(self, frame, prefix):
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        if frame.mean() < 8:              # 幾乎全黑（轉場、黑畫面）不要
            self.skipped += 1
            return
        h = dhash(frame)
        if self.last_hash is not None and np.count_nonzero(h != self.last_hash) < self.min_diff:
            self.skipped += 1             # 跟上一張太像
            return
        self.last_hash = h
        if frame.shape[1] > SAVE_WIDTH:
            s = SAVE_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        name = f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{self.saved:05d}.jpg"
        cv2.imwrite(str(self.out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.saved += 1
        if self.saved % 20 == 0:
            print(f"已存 {self.saved} 張（跳過重複 {self.skipped} 張）")


def from_videos(paths, every, saver):
    for p in paths:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            print(f"[錯誤] 打不開影片：{p}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        print(f"[資訊] {p}：長度 {total / 60:.1f} 分鐘，每 {every} 秒抽一張")
        t = 0.0
        prefix = Path(p).stem[:30]
        while t < total:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                break
            saver.save(frame, prefix)
            t += every
        cap.release()


def from_screen(every, minutes, monitor, saver):
    import dxcam
    camera = dxcam.create(output_idx=monitor, output_color="BGR")
    end = time.time() + minutes * 60
    print(f"[資訊] 開始截圖：每 {every} 秒一張，共 {minutes} 分鐘。按 Ctrl + C 停止")
    try:
        while time.time() < end:
            frame = camera.grab()
            if frame is not None:
                saver.save(frame, "screen")
            time.sleep(every)
    finally:
        del camera


def main():
    ap = argparse.ArgumentParser(description="收集訓練用的畫面")
    ap.add_argument("--video", nargs="+", help="影片檔（可以放好幾個）")
    ap.add_argument("--screen", action="store_true", help="改成截螢幕")
    ap.add_argument("--every", type=float, default=2.0, help="每幾秒取一張（預設 2）")
    ap.add_argument("--minutes", type=float, default=60, help="截螢幕模式：截多久（分鐘）")
    ap.add_argument("--monitor", type=int, default=0, help="截螢幕模式：哪一個螢幕")
    ap.add_argument("--out", default="dataset/raw", help="存放位置（預設 dataset/raw）")
    ap.add_argument("--min-diff", type=int, default=25,
                    help="跟上一張差多少才存（0~256，越大越嚴格，預設 25）")
    args = ap.parse_args()

    if not args.video and not args.screen:
        ap.error("請用 --video 指定影片檔，或加 --screen 截螢幕")

    saver = Saver(args.out, args.min_diff)
    try:
        if args.video:
            from_videos(args.video, args.every, saver)
        if args.screen:
            from_screen(args.every, args.minutes, args.monitor, saver)
    except KeyboardInterrupt:
        pass
    print(f"[完成] 共存了 {saver.saved} 張到 {saver.out}（跳過重複/黑畫面 {saver.skipped} 張）")


if __name__ == "__main__":
    main()

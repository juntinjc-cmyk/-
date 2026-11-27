"""
自動標註：用最大最準的切割模型（老師）幫 dataset/raw/ 裡的每張圖描出「人」的輪廓，
整理成 YOLO 訓練用的資料夾格式。

    python training/auto_label.py

產生：
    dataset/images/train, dataset/images/val   圖片（9:1 分成訓練用 / 考試用）
    dataset/labels/train, dataset/labels/val   每張圖對應的輪廓標註
    dataset/data.yaml                          訓練設定檔

標好之後，可以用標註工具（例如 CVAT）打開 dataset/ 手動修正標錯的地方，效果會更好。
重新執行會覆蓋 dataset/images 和 dataset/labels（dataset/raw 不會動）。
"""

import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="自動標註訓練資料")
    ap.add_argument("--raw", default="dataset/raw", help="原始圖片資料夾")
    ap.add_argument("--out", default="dataset", help="輸出的資料集資料夾")
    ap.add_argument("--teacher", default="yolo11x-seg.pt",
                    help="老師模型（預設 yolo11x-seg.pt，第一次會自動下載）")
    ap.add_argument("--conf", type=float, default=0.35, help="老師的信心門檻（預設 0.35）")
    ap.add_argument("--imgsz", type=int, default=1280, help="老師看圖的尺寸，越大越能抓到小人")
    ap.add_argument("--min-height", type=float, default=0.02,
                    help="比畫面高度的這個比例還小的人不標（太小學不到東西）")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="留多少比例當考試用（預設 0.1）")
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    imgs = sorted(p for p in raw.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not imgs:
        print(f"[錯誤] {raw} 裡面沒有圖片，請先執行 collect_frames.py")
        return

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = out / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    random.seed(0)
    order = imgs[:]
    random.shuffle(order)
    n_val = max(1, int(len(order) * args.val_ratio))
    val_set = set(order[:n_val])

    teacher = YOLO(args.teacher)
    print(f"[資訊] 共 {len(imgs)} 張，訓練 {len(imgs) - n_val} 張 / 考試 {n_val} 張。開始標註…")

    people = empty = 0
    for i, p in enumerate(imgs, 1):
        split = "val" if p in val_set else "train"
        res = teacher.predict(str(p), classes=[0], conf=args.conf, imgsz=args.imgsz,
                              retina_masks=True, verbose=False)[0]
        lines = []
        if res.masks is not None:
            for poly, box in zip(res.masks.xyn, res.boxes.xywhn.cpu().numpy()):
                if len(poly) < 3 or box[3] < args.min_height:
                    continue
                lines.append("0 " + " ".join(f"{v:.5f}" for v in poly.reshape(-1)))
        people += len(lines)
        empty += not lines
        shutil.copy2(p, out / "images" / split / p.name)
        (out / "labels" / split / f"{p.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        if i % 25 == 0 or i == len(imgs):
            print(f"  {i}/{len(imgs)} 張，目前標了 {people} 個人")

    yaml = (f"path: {out.resolve().as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: person\n")
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"[完成] 共標了 {people} 個人；沒有人的圖 {empty} 張（也會用來教模型「這裡沒有人」）")
    print(f"[完成] 設定檔：{out / 'data.yaml'}。下一步：python training/train.py")


if __name__ == "__main__":
    main()

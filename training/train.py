"""
用你的顯卡微調模型 → 比較訓練前後的分數 → 分數比較好才轉成 TensorRT。

    python training/train.py

預設從 yolo11s-seg.pt 開始，訓練 60 輪。1000 張圖在 RTX 5060 上大約 1~2 小時。
訓練時顯卡會跑滿，建議不要同時看影片或玩遊戲。中途可以按 Ctrl + C 停止，
之後加 --resume 從中斷的地方繼續。

完成後會產生（如果分數有變好）：
    yolo11s-seg-custom.engine    直接給 person_overlay.py 用
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

RUN_NAME = "person-custom"


def score(weights, data, imgsz):
    """在考試用的圖片上打分數（輪廓準確度 mAP50-95，0~1，越高越好）。"""
    m = YOLO(weights)
    r = m.val(data=data, imgsz=imgsz, classes=[0], batch=8, plots=False, verbose=False)
    return float(r.seg.map), float(r.seg.map50)


def main():
    ap = argparse.ArgumentParser(description="微調人物切割模型")
    ap.add_argument("--data", default="dataset/data.yaml", help="資料集設定檔")
    ap.add_argument("--model", default="yolo11s-seg.pt", help="從哪個模型開始（預設 yolo11s-seg.pt）")
    ap.add_argument("--epochs", type=int, default=60, help="訓練幾輪（預設 60）")
    ap.add_argument("--imgsz", type=int, default=960, help="訓練尺寸（預設 960）")
    ap.add_argument("--batch", type=int, default=-1, help="一次幾張；-1 自動依顯示記憶體決定")
    ap.add_argument("--export-imgsz", type=int, nargs=2, default=[544, 960],
                    metavar=("H", "W"), help="轉 TensorRT 的尺寸（要跟執行時 --imgsz 一樣）")
    ap.add_argument("--resume", action="store_true", help="從上次中斷的地方繼續訓練")
    ap.add_argument("--force-export", action="store_true", help="分數沒變好也照樣轉檔")
    args = ap.parse_args()

    if not Path(args.data).exists():
        print(f"[錯誤] 找不到 {args.data}，請先執行 training/auto_label.py")
        return

    run_dir = Path("runs") / "segment" / RUN_NAME
    if args.resume:
        model = YOLO(str(run_dir / "weights" / "last.pt"))
        model.train(resume=True)
    else:
        print("[1/3] 先幫原本的模型打分數…")
        base_map, base_map50 = score(args.model, args.data, args.imgsz)
        print(f"      原本：{base_map:.3f}（mAP50 {base_map50:.3f}）")

        print(f"[2/3] 開始訓練 {args.epochs} 輪…")
        model = YOLO(args.model)
        model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    project="runs/segment", name=RUN_NAME, exist_ok=True,
                    patience=15,      # 連續 15 輪沒進步就提早結束
                    workers=4, plots=True)

    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"[錯誤] 找不到訓練結果 {best}")
        return

    print("[3/3] 幫訓練後的模型打分數…")
    new_map, new_map50 = score(str(best), args.data, args.imgsz)
    print(f"      訓練後：{new_map:.3f}（mAP50 {new_map50:.3f}）")
    if not args.resume:
        diff = new_map - base_map
        print(f"      變化：{diff:+.3f}  （{'變好了' if diff > 0 else '沒有變好'}）")
        if diff <= 0 and not args.force_export:
            print("[結束] 分數沒有變好，不轉檔。可以多收集一些畫面、手動修正標註後再試。")
            return

    print("[轉檔] 轉成 TensorRT（要幾分鐘）…")
    engine = YOLO(str(best)).export(format="engine", half=True, imgsz=args.export_imgsz)
    target = Path("yolo11s-seg-custom.engine")
    shutil.copy2(engine, target)
    h, w = args.export_imgsz
    print(f"[完成] {target}\n"
          f"       執行：python person_overlay.py --model {target} --imgsz {h} {w} --recordable")


if __name__ == "__main__":
    main()

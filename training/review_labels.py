"""
用 Gemini 幫自動標註「把關」：刪掉不是真人的標註，列出可能有漏標的圖片。

在 auto_label.py 之後、train.py 之前執行：

    set GEMINI_API_KEY=你的金鑰          （Windows 命令提示字元，只對這個視窗有效）
    python training/review_labels.py

- 每張圖呼叫一次 Gemini：把每個標註畫成有編號的框，問它哪些不是真人、有沒有漏掉的人。
- 會依照 --rpm 自動放慢速度，配合免費額度。
- 進度存在 dataset/review_cache.json。額度用完會自動停下，下次執行從中斷處繼續。
- 第一次修改標註前，會把原本的標註備份到 dataset/labels_backup/。
- 可能有漏標的圖片會列在 dataset/review_missed.txt，你只要手動補這幾張就好。

注意：圖片會上傳給 Google。免費方案的資料可能被 Google 用來改進產品。
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROMPT = """You are checking labels for a person-detection training set.
The image is a frame from a video. Each numbered green box marks something an automatic
detector labeled as a "person".

1. For each number, decide whether the box mainly contains a real human who is physically
   present in the filmed scene (partially visible, dark, blurry or small people still count).
   Mark it NOT a person if it is: a poster/printed photo, a statue, a mannequin, a person shown
   on a TV/monitor/projection screen inside the scene, an on-screen icon or silhouette,
   a reflection, a drawing/cartoon, or just background with no human.
2. Say whether there are clearly visible real people in the scene that have NO box.

Reply with JSON only:
{"not_person": [numbers], "missed_people": true or false, "missed_note": "short description"}"""

VIEW_WIDTH = 1280  # 傳給 Gemini 的圖寬（越小越省額度，但小人會看不清楚）


def read_labels(txt):
    lines = [ln for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    polys = []
    for ln in lines:
        v = np.array(ln.split()[1:], dtype=np.float32).reshape(-1, 2)
        polys.append(v)
    return lines, polys


def render(img_path, polys):
    """把每個標註畫成有編號的框，回傳 JPEG bytes。"""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    s = min(1.0, VIEW_WIDTH / w)
    if s < 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    for i, p in enumerate(polys, 1):
        x1, y1 = (p.min(axis=0) * (w, h)).astype(int)
        x2, y2 = (p.max(axis=0) * (w, h)).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = max(y1, th + 6)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


def ask_gemini(client, types, model, jpg):
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=jpg, mime_type="image/jpeg"), PROMPT],
        config=types.GenerateContentConfig(response_mime_type="application/json",
                                           temperature=0),
    )
    data = json.loads(resp.text)
    return {"not_person": [int(n) for n in data.get("not_person", [])],
            "missed_people": bool(data.get("missed_people", False)),
            "missed_note": str(data.get("missed_note", ""))[:200]}


def is_daily_quota(err):
    s = str(err).lower()
    return "per day" in s or "perday" in s or "daily" in s


def main():
    ap = argparse.ArgumentParser(description="用 Gemini 檢查自動標註")
    ap.add_argument("--dataset", default="dataset", help="資料集資料夾")
    ap.add_argument("--model", default="gemini-flash-latest",
                    help="Gemini 模型名稱（預設 gemini-flash-latest）")
    ap.add_argument("--rpm", type=float, default=8,
                    help="每分鐘最多幾次請求（預設 8，請依你帳號的免費額度調整）")
    ap.add_argument("--limit", type=int, default=0, help="這次最多檢查幾張（0 = 全部）")
    ap.add_argument("--dry-run", action="store_true", help="只檢查、不修改標註")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("[錯誤] 沒有設定 GEMINI_API_KEY。在這個黑色視窗先執行：set GEMINI_API_KEY=你的金鑰")
        return
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("[錯誤] 還沒安裝 Gemini 套件，請先執行：pip install google-genai")
        return
    client = genai.Client(api_key=key)

    ds = Path(args.dataset)
    cache_path = ds / "review_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    items = []
    for split in ("train", "val"):
        for txt in sorted((ds / "labels" / split).glob("*.txt")):
            img = next((ds / "images" / split / f"{txt.stem}{ext}"
                        for ext in (".jpg", ".jpeg", ".png")
                        if (ds / "images" / split / f"{txt.stem}{ext}").exists()), None)
            if img is not None:
                items.append((split, txt, img))
    todo = [it for it in items if f"{it[0]}/{it[1].stem}" not in cache]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[資訊] 共 {len(items)} 張，已檢查 {len(items) - len([i for i in items if f'{i[0]}/{i[1].stem}' not in cache])} 張，"
          f"這次要檢查 {len(todo)} 張（每分鐘最多 {args.rpm:g} 次）")

    interval = 60.0 / args.rpm
    last = 0.0
    done = 0
    try:
        for split, txt, img in todo:
            _, polys = read_labels(txt)
            jpg = render(img, polys)
            for attempt in range(5):
                wait = interval - (time.time() - last)
                if wait > 0:
                    time.sleep(wait)
                last = time.time()
                try:
                    result = ask_gemini(client, types, args.model, jpg)
                    break
                except Exception as e:
                    msg = str(e)
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        if is_daily_quota(e):
                            raise SystemExit("[停止] 今天的免費額度用完了。進度已存檔，明天再執行就會繼續。")
                        time.sleep(30 * (attempt + 1))   # 每分鐘額度：等一下再試
                        continue
                    if isinstance(e, (json.JSONDecodeError, ValueError)):
                        continue                          # 回答格式不對，重問一次
                    raise
            else:
                print(f"  [略過] {img.name}：重試多次仍失敗")
                continue
            result["n"] = len(polys)
            cache[f"{split}/{txt.stem}"] = result
            done += 1
            if done % 10 == 0:
                cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                print(f"  已檢查 {done}/{len(todo)} 張")
    except KeyboardInterrupt:
        print("\n[中斷] 進度已存檔")
    except SystemExit as e:
        print(e)
    finally:
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    # ---- 套用結果 ----
    removed = 0
    missed = []
    backup = ds / "labels_backup"
    for split, txt, img in items:
        r = cache.get(f"{split}/{txt.stem}")
        if r is None:
            continue
        if r["missed_people"]:
            missed.append(f"{img}\t{r['missed_note']}")
        lines, polys = read_labels(txt)
        if r.get("n") != len(lines) or not r["not_person"]:
            continue   # 標註檔在檢查後被改過，或沒有要刪的，就不動
        drop = {n - 1 for n in r["not_person"] if 1 <= n <= len(lines)}
        if not drop or args.dry_run:
            removed += len(drop)
            continue
        dst = backup / split / txt.name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(txt, dst)
        keep = [ln for i, ln in enumerate(lines) if i not in drop]
        txt.write_text("\n".join(keep), encoding="utf-8")
        r["n"] = len(keep)
        r["not_person"] = []
        removed += len(drop)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    (ds / "review_missed.txt").write_text("\n".join(missed), encoding="utf-8")

    action = "會刪除（試跑，沒有真的改）" if args.dry_run else "已刪除"
    print(f"[完成] {action} {removed} 個不是真人的標註；原本的標註備份在 {backup}")
    print(f"[完成] 可能有漏標的圖片 {len(missed)} 張，清單在 {ds / 'review_missed.txt'}")
    remaining = len([i for i in items if f'{i[0]}/{i[1].stem}' not in cache])
    if remaining:
        print(f"[提醒] 還有 {remaining} 張沒檢查，之後再執行一次會繼續。")


if __name__ == "__main__":
    sys.exit(main())

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
    p.add_argument("--no-track", action="store_true", help="關閉追蹤（框可能會閃爍）")
    p.add_argument("--hold", type=int, default=8,
                   help="追蹤：人暫時抓不到時，框最多保留幾幀（預設 8，60 FPS 約 0.13 秒）")
    p.add_argument("--track-low", type=float, default=0.2,
                   help="追蹤：已經在追的人，信心掉到多低還繼續跟（預設 0.2）")
    p.add_argument("--smooth", type=float, default=0.7,
                   help="追蹤：人移動很快時框的跟手程度，0~1，越大越跟手（預設 0.7）")
    p.add_argument("--steady", type=float, default=0.15,
                   help="追蹤：人幾乎沒動時框的跟手程度，0~1，越小越穩、越不抖（預設 0.15）")
    p.add_argument("--min-height", type=float, default=0.0,
                   help="框的高度小於畫面這個比例就不顯示（過濾太遠的人），例如 0.08；預設 0 不過濾")
    p.add_argument("--max-boxes", type=int, default=0,
                   help="最多只顯示幾個框（留最大、最近的），例如 10；預設 0 不限制")
    p.add_argument("--ignore", type=float, nargs=4, action="append",
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="忽略區域（用畫面比例 0~1 表示）：框的中心落在這裡就不顯示。可以寫多次。"
                        "預設是左下角 0 0.65 0.11 1")
    p.add_argument("--no-ignore", action="store_true", help="不使用任何忽略區域")
    p.add_argument("--keep-self", action="store_true",
                   help="不要過濾第一人稱畫面下方的手（預設會過濾）")
    p.add_argument("--self-top", type=float, default=0.5,
                   help="過濾手的靈敏度：碰到畫面底部、且頂端低於畫面這個高度比例的框會被當成自己的手。"
                        "預設 0.5（畫面一半）；手還是被框到就調小，例如 0.35")
    p.add_argument("--cpu-preprocess", action="store_true",
                   help="改用 ultralytics 原本的完整流程（較慢；快速流程有問題時才用）")
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
    for x1, y1, x2, y2 in boxes:  # 只畫框，不畫文字
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


def filter_self(xyxy, confs, h, w, self_top):
    """第一人稱畫面：去掉拍攝者自己露出來的手、手臂。

    判斷方式：框碰到畫面最底部，而且框的頂端在畫面下方（低於 self_top 的高度比例）。
    真正站在前面的人，框通常會從畫面中上方開始，不會被誤刪。
    """
    if len(xyxy) == 0:
        return xyxy, confs
    touches_bottom = xyxy[:, 3] >= h * 0.97
    starts_low = xyxy[:, 1] >= h * self_top
    keep = ~(touches_bottom & starts_low)
    return xyxy[keep], confs[keep]


def iou_matrix(a, b):
    """a: (N,4), b: (M,4) 的 xyxy 框，回傳 (N,M) 的 IoU（重疊比例）。"""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


class BoxTracker:
    """簡單的追蹤器，讓框在快速運動時不閃爍、不跳動，也不要太雜亂。

    - 雙門檻：新出現的人要信心 >= high 才開始追；已經在追蹤的人，
      畫面糊掉分數變低（>= track_low）也繼續跟著。
    - 新的人要連續被看到 min_hits 次才顯示，過濾只出現一瞬間的誤判。
    - 暫時抓不到時，框短暫沿原方向移動後就停住，最多撐 hold 幀才消失（不會到處飄）。
    - 跟現有框幾乎重疊的偵測，不會再開一個新框（避免同一個人兩個框）。
    - 位置做平滑處理，減少每幀的小抖動；smooth 越大越跟手、越小越穩。
    """

    def __init__(self, high=0.4, hold=8, smooth=0.7, steady=0.15, match_iou=0.2, min_hits=2,
                 dup_iou=0.5):
        self.high = high
        self.hold = hold
        self.smooth = smooth    # 移動很快時的跟手程度
        self.steady = steady    # 幾乎沒動時的跟手程度（越小越穩，越不抖）
        self.match_iou = match_iou
        self.min_hits = min_hits
        self.dup_iou = dup_iou
        self.boxes = np.zeros((0, 4), dtype=np.float32)
        self.vel = np.zeros((0, 4), dtype=np.float32)
        self.conf = np.zeros((0,), dtype=np.float32)
        self.misses = np.zeros((0,), dtype=np.int32)
        self.hits = np.zeros((0,), dtype=np.int32)

    def update(self, det, det_conf, h, w):
        det = det.astype(np.float32)
        pred = self.boxes + self.vel  # 依照速度預測這一幀的位置（快速運動時比較對得上）

        matched_t, matched_d = [], []
        if len(pred) and len(det):
            iou = iou_matrix(pred, det)
            # 快速運動時框可能幾乎不重疊，所以也看「中心點距離 / 框的大小」
            pc = (pred[:, :2] + pred[:, 2:]) / 2
            dc = (det[:, :2] + det[:, 2:]) / 2
            size = np.maximum(pred[:, 2] - pred[:, 0], pred[:, 3] - pred[:, 1])[:, None] + 1e-6
            dist = np.linalg.norm(pc[:, None, :] - dc[None, :, :], axis=2) / size
            ok = (iou >= self.match_iou) | (dist <= 0.5)
            score = np.where(ok, iou - dist, -np.inf)
            # 由最像的開始配對，每個追蹤對象和偵測框各只配一次
            for flat in np.argsort(-score, axis=None):
                t, d = divmod(int(flat), score.shape[1])
                if not np.isfinite(score[t, d]):
                    break
                if t in matched_t or d in matched_d:
                    continue
                matched_t.append(t)
                matched_d.append(d)

        for t, d in zip(matched_t, matched_d):
            # 自動調整平滑程度：偵測框只偏一點點（模型本身的小誤差）→ 幾乎不動，避免抖動；
            # 偏很多（人真的在動）→ 快速跟上。
            size = max(pred[t, 2] - pred[t, 0], pred[t, 3] - pred[t, 1], 1.0)
            move = np.abs(det[d] - pred[t]).mean() / size
            k = np.clip((move - 0.01) / 0.08, 0.0, 1.0)
            a = self.steady + (self.smooth - self.steady) * k
            new = a * det[d] + (1 - a) * pred[t]
            self.vel[t] = 0.7 * self.vel[t] + 0.3 * (new - self.boxes[t])
            self.boxes[t] = new
            self.conf[t] = det_conf[d]
            self.misses[t] = 0
            self.hits[t] += 1

        # 沒配到的追蹤對象：沿原方向短暫移動，速度很快歸零（停住，不會飄走）
        lost = np.ones(len(self.boxes), dtype=bool)
        lost[matched_t] = False
        self.boxes[lost] = pred[lost]
        self.vel[lost] *= 0.5
        self.misses[lost] += 1

        # 沒配到的偵測框：信心夠高、而且沒有跟現有的框大幅重疊，才當成新的人
        cand = [d for d in range(len(det)) if d not in matched_d and det_conf[d] >= self.high]
        if cand and len(self.boxes):
            overlap = iou_matrix(det[cand], self.boxes).max(axis=1)
            cand = [d for d, o in zip(cand, overlap) if o < self.dup_iou]
        if cand:
            n = len(cand)
            self.boxes = np.concatenate([self.boxes, det[cand]])
            self.vel = np.concatenate([self.vel, np.zeros((n, 4), np.float32)])
            self.conf = np.concatenate([self.conf, det_conf[cand].astype(np.float32)])
            self.misses = np.concatenate([self.misses, np.zeros(n, np.int32)])
            self.hits = np.concatenate([self.hits, np.ones(n, np.int32)])

        # 太久沒抓到、或已經跑出畫面的就刪掉
        b = self.boxes
        inside = (b[:, 2] > 0) & (b[:, 0] < w) & (b[:, 3] > 0) & (b[:, 1] < h)
        alive = (self.misses <= self.hold) & inside
        self._keep(alive)

        # 兩個追蹤框幾乎重疊（同一個人被追了兩次）：留下比較穩定的那個
        if len(self.boxes) > 1:
            order = np.lexsort((-self.hits, self.misses))  # 先看最近有沒有抓到，再看追了多久
            iou = iou_matrix(self.boxes, self.boxes)
            keep = np.ones(len(self.boxes), dtype=bool)
            for i_pos, i in enumerate(order):
                if not keep[i]:
                    continue
                for j in order[i_pos + 1:]:
                    if keep[j] and iou[i, j] > self.dup_iou:
                        keep[j] = False
            self._keep(keep)

        show = self.hits >= self.min_hits
        out = self.boxes[show].copy()
        out[:, 0::2] = np.clip(out[:, 0::2], 0, w - 1)
        out[:, 1::2] = np.clip(out[:, 1::2], 0, h - 1)
        return out.round().astype(np.int32), self.conf[show].copy()

    def _keep(self, mask):
        self.boxes, self.vel = self.boxes[mask], self.vel[mask]
        self.conf, self.misses, self.hits = self.conf[mask], self.misses[mask], self.hits[mask]


def limit_boxes(xyxy, confs, h, min_height, max_boxes):
    """過濾太小（太遠）的框，以及只保留最大的幾個框，讓畫面不要太雜。"""
    if len(xyxy) == 0:
        return xyxy, confs
    if min_height > 0:
        keep = (xyxy[:, 3] - xyxy[:, 1]) >= h * min_height
        xyxy, confs = xyxy[keep], confs[keep]
    if max_boxes > 0 and len(xyxy) > max_boxes:
        area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        idx = np.argsort(-area)[:max_boxes]
        xyxy, confs = xyxy[idx], confs[idx]
    return xyxy, confs


DEFAULT_IGNORE = [(0.0, 0.65, 0.11, 1.0)]  # 左下角（使用者指定的範圍）


def filter_ignore(xyxy, confs, h, w, regions):
    """框的中心落在任何一個忽略區域（畫面比例）裡，就拿掉。"""
    if len(xyxy) == 0 or not regions:
        return xyxy, confs
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2 / w
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2 / h
    drop = np.zeros(len(xyxy), dtype=bool)
    for x1, y1, x2, y2 in regions:
        drop |= (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
    return xyxy[~drop], confs[~drop]


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

        x = g[..., :3].permute(2, 0, 1).flip(0).float().div_(255.0)  # HWC BGR(A) -> CHW RGB, 0~1
        canvas = torch.full((1, 3, self.th, self.tw), 114 / 255.0,
                            device=self.device, dtype=torch.float32)  # 灰色補邊，跟 ultralytics 一樣
        canvas[0, :, :nh, :nw] = x
        return canvas, r


def nms_numpy(boxes, scores, iou_thr):
    """標準的非極大值抑制：重疊太多的框只留分數最高的那個。回傳保留的索引。"""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


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
        # 有追蹤時，模型先用較低的門檻抓「可能是人」的框，再交給追蹤器決定要不要顯示
        det_conf = args.conf if args.no_track else min(args.conf, args.track_low)
        self.kwargs = dict(classes=[PERSON_CLASS_ID], conf=det_conf, imgsz=imgsz,
                           device=device, verbose=False)
        if gpu and not args.model.endswith(".engine"):
            # .engine 轉換時已經是 FP16，不用再指定。
            # 新版 ultralytics 把 half 改名成 quantize，舊版沒有 quantize，兩種都相容
            from ultralytics.cfg import DEFAULT_CFG_DICT
            if "quantize" in DEFAULT_CFG_DICT:
                self.kwargs["quantize"] = 16
            else:
                self.kwargs["half"] = True

        self.conf = det_conf
        self.tracker = None if args.no_track else BoxTracker(
            high=args.conf, hold=args.hold, smooth=args.smooth, steady=args.steady)
        self.keep_self = args.keep_self
        self.min_height = args.min_height
        self.max_boxes = args.max_boxes
        self.ignore = [] if args.no_ignore else (args.ignore or DEFAULT_IGNORE)
        if self.ignore:
            print(f"[資訊] 忽略區域（畫面比例）: {self.ignore}")
        self.self_top = args.self_top
        self.iou = 0.7          # 跟 ultralytics 預設一樣
        self.max_det = 300

        # 暖機：先用 ultralytics 原本的流程跑幾次，順便讓它把模型（AutoBackend）準備好
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(3):
            self._predict_ultralytics(dummy)

        # 快速流程：自己做前處理 + 直接呼叫模型 + 只針對「人」的精簡後處理。
        # ultralytics 的後處理要處理 80 個類別、建立 Results 物件，實測要 8~10 ms，
        # 精簡版只看「人」這一類，全部在 GPU 上做，通常不到 1 ms。
        self.fast = False
        if gpu and not args.cpu_preprocess:
            try:
                self.backend = self.model.predictor.model
                self.in_dtype = torch.float16 if getattr(self.backend, "fp16", False) else torch.float32
                self.pre = GpuPreprocessor((h, w), device)
                self.fast = True
                for _ in range(3):
                    self._predict_fast(dummy)
            except Exception as e:  # 任何問題就退回 ultralytics 原本的流程
                self.fast = False
                print(f"[警告] 快速流程無法使用，改用 ultralytics 原本的流程：{e!r}")
        print(f"[資訊] 偵測流程: {'快速（GPU 前處理 + 精簡後處理）' if self.fast else 'ultralytics 原本的流程'}")

    def _predict_ultralytics(self, frame):
        res = self.model.predict(frame, **self.kwargs)[0]
        self.speed = res.speed
        return res.boxes.xyxy.cpu().numpy().astype(int), res.boxes.conf.cpu().numpy()

    def _predict_fast(self, frame):
        t0 = time.perf_counter()
        x, r = self.pre(frame)
        x = x.to(self.in_dtype)
        t1 = time.perf_counter()

        out = self.backend(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = out[0]                       # (4 + 類別數, 候選框數)
        nc = len(self.model.names)
        if out.shape[0] != 4 + nc:
            if out.shape[-1] == 4 + nc:
                out = out.T
            else:
                raise RuntimeError(f"不支援的模型輸出形狀 {tuple(out.shape)}")
        # 只取 cx, cy, w, h 和「人」的分數這 5 列，一次搬回 CPU。
        # 在 GPU 上做很多小步驟，每一步都要等顯卡（顯卡同時在播影片），反而比較慢。
        sub = out[[0, 1, 2, 3, 4 + PERSON_CLASS_ID]].float().cpu().numpy()
        t2 = time.perf_counter()

        scores = sub[4]
        keep = scores > self.conf
        if not keep.any():
            xyxy, conf = np.zeros((0, 4), dtype=np.int32), np.zeros((0,), dtype=np.float32)
        else:
            cx, cy, bw, bh = sub[0, keep], sub[1, keep], sub[2, keep], sub[3, keep]
            sc = scores[keep]
            boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
            idx = nms_numpy(boxes, sc, self.iou)[: self.max_det]
            boxes = boxes[idx] / r         # 換回原始畫面的座標
            fh, fw = frame.shape[:2]
            np.clip(boxes[:, 0::2], 0, fw - 1, out=boxes[:, 0::2])
            np.clip(boxes[:, 1::2], 0, fh - 1, out=boxes[:, 1::2])
            xyxy = boxes.round().astype(np.int32)
            conf = sc[idx]
        t3 = time.perf_counter()
        # 推論包含把結果搬回 CPU（要等顯卡算完）
        self.speed = {"preprocess": (t1 - t0) * 1000, "inference": (t2 - t1) * 1000,
                      "postprocess": (t3 - t2) * 1000}
        return xyxy, conf

    def __call__(self, frame):
        """回傳 (框 xyxy 整數陣列, 信心分數陣列)。self.speed 記錄前處理 / 推論 / 後處理毫秒數。"""
        xyxy, confs = self._predict_fast(frame) if self.fast else self._predict_ultralytics(frame)
        xyxy, confs = filter_ignore(xyxy, confs, frame.shape[0], frame.shape[1], self.ignore)
        if not self.keep_self:
            xyxy, confs = filter_self(xyxy, confs, frame.shape[0], frame.shape[1], self.self_top)
        if self.tracker is not None:
            xyxy, confs = self.tracker.update(xyxy, confs, frame.shape[0], frame.shape[1])
        xyxy, confs = limit_boxes(xyxy, confs, frame.shape[0], self.min_height, self.max_boxes)
        return xyxy, confs


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

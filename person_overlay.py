"""
螢幕疊加版人物偵測：不開預覽視窗，綠框直接畫在螢幕上（例如看 YouTube 影片時）。

- 透明、置頂、滑鼠點得穿的視窗蓋在整個螢幕上，只畫框。
- 偵測和疊加視窗分成兩個獨立的程式（process）跑，互不拖慢：
  主程式負責擷取 + 偵測，子程式只負責畫框，兩者用 Pipe 傳框的座標。
- 透明的做法是「色鍵」：視窗裡純黑色的像素會被 Windows 當成透明。
  每幀只重畫框的邊線那幾條細長區域，不用整個 4K 畫面搬來搬去，所以很省。
- 預設：疊加視窗會被排除在螢幕擷取之外，所以框不會被下一幀再偵測到，
  但錄影軟體（OBS、剪取工具）也錄不到框。
- 加 --recordable：錄影軟體錄得到框。代價是框也會被 dxcam 擷取進偵測畫面，
  通常不影響偵測人物。
- 疊加畫面由 Windows 桌面合成器（DWM）在顯示卡上合成。

用法：
    python person_overlay.py --model yolo11n.engine
    python person_overlay.py --model yolo11n.engine --recordable   # 要錄影時

結束：回到終端機按 Ctrl + C。
"""

import ctypes
import multiprocessing as mp
import signal
import sys
import time
from ctypes import wintypes

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen, QRegion
from PySide6.QtWidgets import QApplication, QWidget

# 偵測相關的套件（torch、ultralytics、dxcam）只在主程式載入，
# 疊加視窗的子程式不需要，載入會白白多花好幾秒和記憶體。

BOX_COLOR = QColor(0, 255, 0)
KEY_COLOR = QColor(0, 0, 0)          # 純黑 = 透明，畫面上其他地方都不能用純黑
LABEL_TEXT_COLOR = QColor(12, 12, 12)
HUD_BG_COLOR = QColor(24, 24, 24)
HUD_INTERVAL = 0.25                  # 左上角 FPS 每 0.25 秒更新一次就夠了

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
LWA_COLORKEY = 0x1


def apply_color_key(hwnd):
    """把視窗設成「純黑色 = 透明」、滑鼠點得穿。成功回傳 True。"""
    user32 = ctypes.windll.user32
    user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    ex = user32.GetWindowLongPtrW(wintypes.HWND(hwnd), GWL_EXSTYLE)
    ex |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
    user32.SetWindowLongPtrW(wintypes.HWND(hwnd), GWL_EXSTYLE, ex)
    if not user32.SetLayeredWindowAttributes(wintypes.HWND(hwnd), 0x000000, 0, LWA_COLORKEY):
        return False
    # 確認真的套用成功，避免整個螢幕被黑色蓋住
    key = wintypes.DWORD()
    alpha = ctypes.c_ubyte()
    flags = wintypes.DWORD()
    ok = user32.GetLayeredWindowAttributes(wintypes.HWND(hwnd), ctypes.byref(key),
                                           ctypes.byref(alpha), ctypes.byref(flags))
    return bool(ok) and bool(flags.value & LWA_COLORKEY) and key.value == 0


def run_detection(args, conn):
    """主程式：擷取螢幕 → 偵測 → 把框（螢幕實體像素座標）送給疊加視窗子程式。"""
    import dxcam
    from person_detector import Detector, compute_region

    detector = Detector(args)
    camera = dxcam.create(output_idx=args.monitor, output_color="BGR")
    region = compute_region(args, camera.width, camera.height)
    left, top = region[0], region[1]
    print(f"[資訊] 螢幕 {camera.width}x{camera.height}，擷取區域 {region}，目標 {args.fps} FPS")
    camera.start(region=region, target_fps=args.fps, video_mode=True)
    print("[資訊] 開始偵測。要結束請在這個視窗按 Ctrl + C")

    fps_ema = 0.0
    last = last_print = time.perf_counter()
    t_cap = t_inf = 0.0
    n = 0
    try:
        while True:
            t0 = time.perf_counter()
            frame = camera.get_latest_frame()
            if frame is None:
                continue
            t1 = time.perf_counter()
            xyxy, confs = detector(frame)
            boxes = [(int(x1) + left, int(y1) + top, int(x2) + left, int(y2) + top, float(c))
                     for (x1, y1, x2, y2), c in zip(xyxy, confs)]
            now = time.perf_counter()
            t_cap += t1 - t0
            t_inf += now - t1
            n += 1
            dt = now - last
            last = now
            fps_ema = (1.0 / dt) if fps_ema == 0 else fps_ema * 0.9 + (1.0 / dt) * 0.1
            try:
                conn.send((boxes, fps_ema))
            except (BrokenPipeError, OSError):
                print("[資訊] 疊加視窗已關閉")
                break
            if now - last_print >= 1.0:
                print(f"FPS {fps_ema:5.1f} | 擷取等待 {t_cap / n * 1000:5.1f} ms"
                      f" | 偵測 {t_inf / n * 1000:5.1f} ms | 人數 {len(boxes)}")
                t_cap = t_inf = 0.0
                n = 0
                last_print = now
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        del camera


class Overlay(QWidget):
    def __init__(self, screen):
        super().__init__(None,
                         Qt.FramelessWindowHint
                         | Qt.WindowStaysOnTopHint
                         | Qt.Tool                        # 不出現在工作列
                         | Qt.WindowTransparentForInput)  # 滑鼠點得穿
        # 不用 WA_TranslucentBackground（每幀要把整個 4K 畫面送給 Windows，很慢），
        # 改用色鍵：自己把背景畫成純黑，Windows 會把純黑當成透明。
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setGeometry(screen.geometry())
        self.boxes = []
        self.fps = -1.0
        self.hud_fps = -1.0
        self._hud_time = 0.0
        self._dirty = QRegion()

    HUD_RECT = (20, 20, 20 + 700, 20 + 70)  # 左上角 FPS 區塊（實體像素）

    def _to_logical(self, x1, y1, x2, y2, pad=6):
        dpr = self.devicePixelRatioF()
        return QRect(int((x1 - pad) / dpr), int((y1 - pad) / dpr),
                     int((x2 - x1 + 2 * pad) / dpr) + 2, int((y2 - y1 + 2 * pad) / dpr) + 2)

    def _region_for(self, boxes):
        """只算框的四條邊和標籤（不是整個框的內部），換成 Qt 的邏輯座標。"""
        label_h = 40
        region = QRegion()
        for x1, y1, x2, y2, _ in boxes:
            region += self._to_logical(x1, y1 - label_h, max(x2, x1 + 260), y1 + label_h)  # 上邊 + 標籤
            region += self._to_logical(x1, y2, x2, y2)                            # 下邊
            region += self._to_logical(x1, y1, x1, y2)                            # 左邊
            region += self._to_logical(x2, y1, x2, y2)                            # 右邊
        return region

    def set_boxes(self, boxes, fps):
        self.boxes = boxes
        self.fps = fps
        # 只重畫「舊的框 + 新的框」的邊線
        new_dirty = self._region_for(boxes)
        dirty = self._dirty + new_dirty
        now = time.perf_counter()
        if fps < 0 or now - self._hud_time >= HUD_INTERVAL:
            self.hud_fps = fps
            self._hud_time = now
            dirty += self._to_logical(*self.HUD_RECT)
        self.update(dirty)
        self._dirty = new_dirty

    def paintEvent(self, event):
        p = QPainter(self)
        # 先把要重畫的範圍塗成純黑（= 透明），清掉上一幀的框
        p.fillRect(event.rect(), KEY_COLOR)
        # dxcam 給的是實體像素；Windows 有縮放（例如 4K 開 150%）時，Qt 用的是縮放後的座標。
        # 這裡把畫布縮回實體像素，框就會對準。
        dpr = self.devicePixelRatioF()
        p.scale(1.0 / dpr, 1.0 / dpr)

        p.setPen(QPen(BOX_COLOR, 3))
        font = QFont()
        font.setPixelSize(22)
        p.setFont(font)
        fm = p.fontMetrics()

        for x1, y1, x2, y2, c in self.boxes:
            p.setPen(QPen(BOX_COLOR, 3))
            p.setBrush(Qt.NoBrush)
            p.drawRect(x1, y1, x2 - x1, y2 - y1)

            label = f"person {c:.2f}"
            tw, th = fm.horizontalAdvance(label) + 8, fm.height() + 4
            ty = max(y1 - th, 0)
            p.fillRect(QRect(x1, ty, tw, th), BOX_COLOR)
            p.setPen(LABEL_TEXT_COLOR)
            p.drawText(QRect(x1, ty, tw, th), Qt.AlignCenter, label)

        # 左上角顯示 FPS 和人數（fps < 0 代表已停止，不顯示）
        if self.hud_fps >= 0:
            hud = f"FPS {self.hud_fps:5.1f}  |  persons {len(self.boxes)}"
            font.setPixelSize(32)
            font.setBold(True)
            p.setFont(font)
            hm = p.fontMetrics()
            rect = QRect(20, 20, hm.horizontalAdvance(hud) + 24, hm.height() + 12)
            p.fillRect(rect, HUD_BG_COLOR)
            p.setPen(BOX_COLOR)
            p.drawText(rect, Qt.AlignCenter, hud)
        p.end()


def pick_screen(index):
    if index == 0:
        return QGuiApplication.primaryScreen()
    screens = QGuiApplication.screens()
    return screens[index] if index < len(screens) else QGuiApplication.primaryScreen()


def exclude_hwnd_from_capture(hwnd):
    """讓視窗不被螢幕擷取（需要 Windows 10 2004 以上）。"""
    WDA_EXCLUDEFROMCAPTURE = 0x00000011
    if not ctypes.windll.user32.SetWindowDisplayAffinity(wintypes.HWND(hwnd), WDA_EXCLUDEFROMCAPTURE):
        print("[警告] 排除擷取失敗，框可能會被擷取到")


def overlay_process(conn, monitor, recordable, ready):
    """子程式：只負責透明疊加視窗，從 Pipe 收框來畫。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl + C 交給主程式處理

    app = QApplication(sys.argv)
    overlay = Overlay(pick_screen(monitor))
    hwnd = int(overlay.winId())  # 先建立視窗（還不顯示）
    if not apply_color_key(hwnd):
        print("[錯誤] 無法設定透明疊加視窗，程式結束")
        ready.set()
        return
    overlay.show()
    if not apply_color_key(hwnd):  # 顯示後再確認一次，失敗就立刻關掉，避免黑畫面蓋住螢幕
        overlay.close()
        print("[錯誤] 無法設定透明疊加視窗，程式結束")
        ready.set()
        return
    if recordable:
        print("[資訊] 可錄影模式：錄影軟體錄得到框")
    else:
        exclude_hwnd_from_capture(hwnd)
    ready.set()

    def poll():
        latest = None
        try:
            while conn.poll():  # 一次把累積的訊息讀完，只畫最新的一筆
                latest = conn.recv()
                if latest is None:
                    app.quit()
                    return
        except (EOFError, OSError):
            app.quit()
            return
        if latest is not None:
            overlay.set_boxes(*latest)

    timer = QTimer()
    timer.timeout.connect(poll)
    timer.start(2)
    app.exec()


def main():
    from person_detector import build_parser

    parser = build_parser()
    parser.add_argument("--recordable", action="store_true",
                        help="讓錄影軟體也錄得到框（框也會被偵測程式看到）")
    args = parser.parse_args()

    recv_conn, send_conn = mp.Pipe(duplex=False)
    ready = mp.Event()
    proc = mp.Process(target=overlay_process,
                      args=(recv_conn, args.monitor, args.recordable, ready), daemon=True)
    proc.start()
    ready.wait(timeout=30)
    if not proc.is_alive():
        print("[錯誤] 疊加視窗沒有啟動成功")
        return

    try:
        run_detection(args, send_conn)
    finally:
        try:
            send_conn.send(None)  # 通知疊加視窗關閉
        except (BrokenPipeError, OSError):
            pass
        proc.join(timeout=3)
        if proc.is_alive():
            proc.terminate()


if __name__ == "__main__":
    main()

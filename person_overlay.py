"""
螢幕疊加版人物偵測：不開預覽視窗，綠框直接畫在螢幕上（例如看 YouTube 影片時）。

- 透明、置頂、滑鼠點得穿的視窗蓋在整個螢幕上，只畫框。
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

import signal
import sys
import time

import dxcam
from PySide6.QtCore import QRect, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen, QRegion
from PySide6.QtWidgets import QApplication, QWidget

from person_detector import build_parser, Detector, compute_region, exclude_hwnd_from_capture

BOX_COLOR = QColor(0, 255, 0)
LABEL_TEXT_COLOR = QColor(0, 0, 0)


class DetectorThread(QThread):
    """背景執行緒：擷取螢幕 → 偵測 → 把框（螢幕實體像素座標）送給疊加視窗。"""

    detections = Signal(object, float)  # (list[(x1, y1, x2, y2, conf)], fps)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        args = self.args
        # 模型和擷取都在這個執行緒裡建立，避免跨執行緒使用 CUDA / TensorRT
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
            while self._running:
                t0 = time.perf_counter()
                frame = camera.get_latest_frame()
                if frame is None:
                    continue
                t1 = time.perf_counter()
                xyxy, confs = detector(frame)
                boxes = [(x1 + left, y1 + top, x2 + left, y2 + top, float(c))
                         for (x1, y1, x2, y2), c in zip(xyxy, confs)]
                now = time.perf_counter()
                t_cap += t1 - t0
                t_inf += now - t1
                n += 1
                dt = now - last
                last = now
                fps_ema = (1.0 / dt) if fps_ema == 0 else fps_ema * 0.9 + (1.0 / dt) * 0.1
                self.detections.emit(boxes, fps_ema)
                if now - last_print >= 1.0:
                    print(f"FPS {fps_ema:5.1f} | 擷取等待 {t_cap / n * 1000:5.1f} ms"
                          f" | 偵測 {t_inf / n * 1000:5.1f} ms | 人數 {len(boxes)}")
                    t_cap = t_inf = 0.0
                    n = 0
                    last_print = now
        finally:
            camera.stop()
            del camera
            # 清掉螢幕上殘留的框
            self.detections.emit([], -1.0)


class Overlay(QWidget):
    def __init__(self, screen):
        super().__init__(None,
                         Qt.FramelessWindowHint
                         | Qt.WindowStaysOnTopHint
                         | Qt.Tool                        # 不出現在工作列
                         | Qt.WindowTransparentForInput)  # 滑鼠點得穿
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setGeometry(screen.geometry())
        self.boxes = []
        self.fps = -1.0
        self._dirty = QRegion()

    def _region_for(self, boxes):
        """算出這些框（含標籤）和左上角 FPS 佔用的範圍，換成 Qt 的邏輯座標。"""
        dpr = self.devicePixelRatioF()
        pad = 6
        label_h = 40
        region = QRegion()
        rects = [(20, 20, 20 + 700, 20 + 70)]  # 左上角 FPS 區塊
        rects += [(x1, y1 - label_h, max(x2, x1 + 260), y2) for x1, y1, x2, y2, _ in boxes]
        for x1, y1, x2, y2 in rects:
            region += QRect(int((x1 - pad) / dpr), int((y1 - pad) / dpr),
                            int((x2 - x1 + 2 * pad) / dpr) + 2, int((y2 - y1 + 2 * pad) / dpr) + 2)
        return region

    def set_boxes(self, boxes, fps):
        self.boxes = boxes
        self.fps = fps
        # 只重畫「舊的框 + 新的框」所在的範圍，不要每次重畫整個 4K 螢幕，
        # 否則透明全螢幕視窗每幀都要搬 30 多 MB 的畫面，會拖慢速度。
        new_dirty = self._region_for(boxes)
        self.update(self._dirty + new_dirty)
        self._dirty = new_dirty

    def paintEvent(self, _event):
        p = QPainter(self)
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
        if self.fps >= 0:
            hud = f"FPS {self.fps:5.1f}  |  persons {len(self.boxes)}"
            font.setPixelSize(32)
            font.setBold(True)
            p.setFont(font)
            hm = p.fontMetrics()
            rect = QRect(20, 20, hm.horizontalAdvance(hud) + 24, hm.height() + 12)
            p.fillRect(rect, QColor(0, 0, 0, 160))
            p.setPen(BOX_COLOR)
            p.drawText(rect, Qt.AlignCenter, hud)
        p.end()


def pick_screen(index):
    if index == 0:
        return QGuiApplication.primaryScreen()
    screens = QGuiApplication.screens()
    return screens[index] if index < len(screens) else QGuiApplication.primaryScreen()


def main():
    parser = build_parser()
    parser.add_argument("--recordable", action="store_true",
                        help="讓錄影軟體也錄得到框（框也會被偵測程式看到）")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    overlay = Overlay(pick_screen(args.monitor))
    overlay.show()
    if args.recordable:
        print("[資訊] 可錄影模式：錄影軟體錄得到框")
    else:
        exclude_hwnd_from_capture(int(overlay.winId()))

    worker = DetectorThread(args)
    worker.detections.connect(overlay.set_boxes)  # 跨執行緒，Qt 會自動排到主執行緒執行
    worker.finished.connect(app.quit)

    def shutdown(*_):
        worker.stop()

    signal.signal(signal.SIGINT, shutdown)
    # Qt 事件迴圈執行時 Python 收不到 Ctrl + C，用計時器讓它定期醒來處理
    keepalive = QTimer()
    keepalive.timeout.connect(lambda: None)
    keepalive.start(200)

    worker.start()
    app.exec()
    worker.stop()
    worker.wait()


if __name__ == "__main__":
    main()

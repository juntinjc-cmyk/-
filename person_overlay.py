"""
螢幕疊加版人物偵測：不開預覽視窗，綠框直接畫在螢幕上（例如看 YouTube 影片時）。

- 透明、置頂、滑鼠點得穿的視窗蓋在整個螢幕上，只畫框。
- 疊加視窗會被排除在螢幕擷取之外，所以框不會被下一幀再偵測到。
- 疊加畫面由 Windows 桌面合成器（DWM）在顯示卡上合成。

用法：
    python person_overlay.py --model yolo11n.engine

結束：回到終端機按 Ctrl + C。
"""

import signal
import sys
import time

import dxcam
from PySide6.QtCore import QRect, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from person_detector import build_parser, compute_region, exclude_hwnd_from_capture, load_model

BOX_COLOR = QColor(0, 255, 0)
LABEL_TEXT_COLOR = QColor(0, 0, 0)


class DetectorThread(QThread):
    """背景執行緒：擷取螢幕 → 偵測 → 把框（螢幕實體像素座標）送給疊加視窗。"""

    detections = Signal(object)  # list[(x1, y1, x2, y2, conf)]

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        args = self.args
        # 模型和擷取都在這個執行緒裡建立，避免跨執行緒使用 CUDA / TensorRT
        model, pred_kwargs = load_model(args)
        camera = dxcam.create(output_idx=args.monitor, output_color="BGR")
        region = compute_region(args, camera.width, camera.height)
        left, top = region[0], region[1]
        print(f"[資訊] 螢幕 {camera.width}x{camera.height}，擷取區域 {region}，目標 {args.fps} FPS")
        camera.start(region=region, target_fps=args.fps, video_mode=True)
        print("[資訊] 開始偵測。要結束請在這個視窗按 Ctrl + C")

        fps_ema = 0.0
        last = last_print = time.perf_counter()
        try:
            while self._running:
                frame = camera.get_latest_frame()
                if frame is None:
                    continue
                result = model.predict(frame, **pred_kwargs)[0]
                xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                boxes = [(x1 + left, y1 + top, x2 + left, y2 + top, float(c))
                         for (x1, y1, x2, y2), c in zip(xyxy, confs)]
                self.detections.emit(boxes)

                now = time.perf_counter()
                dt = now - last
                last = now
                fps_ema = (1.0 / dt) if fps_ema == 0 else fps_ema * 0.9 + (1.0 / dt) * 0.1
                if now - last_print >= 1.0:
                    print(f"FPS {fps_ema:5.1f} | 人數 {len(boxes)}")
                    last_print = now
        finally:
            camera.stop()
            del camera
            # 清掉螢幕上殘留的框
            self.detections.emit([])


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

    def set_boxes(self, boxes):
        self.boxes = boxes
        self.update()  # 觸發重畫

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
        p.end()


def pick_screen(index):
    if index == 0:
        return QGuiApplication.primaryScreen()
    screens = QGuiApplication.screens()
    return screens[index] if index < len(screens) else QGuiApplication.primaryScreen()


def main():
    args = build_parser().parse_args()

    app = QApplication(sys.argv)
    overlay = Overlay(pick_screen(args.monitor))
    overlay.show()
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

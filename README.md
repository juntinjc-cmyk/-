# 即時人物偵測（dxcam + YOLO）

用 [dxcam](https://github.com/ra1nty/DXcam) 高速擷取螢幕畫面，用 [Ultralytics YOLO](https://docs.ultralytics.com/) 偵測畫面中的人物並畫框，目標 60 FPS。

> dxcam 只支援 **Windows 10/11**。要跑到 60 FPS 需要 **NVIDIA 顯示卡 + CUDA 版 PyTorch**。

## 安裝

```bash
# 1. 先裝 CUDA 版 PyTorch
#    RTX 50 系列（5060/5070/5080/5090，Blackwell 架構）一定要用 CUDA 12.8 以上的版本（cu128）
#    舊版（cu121、cu124）會出現 "no kernel image is available for execution on the device"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. 其他套件
pip install -r requirements.txt
```

## 執行

```bash
python person_detector.py
```

第一次執行會自動下載 `yolo11n.pt` 模型。按 `q` 或 `Esc` 結束。

| 參數 | 預設 | 說明 |
|---|---|---|
| `--size` | 640 | 擷取螢幕中央 N×N 區域，`0` = 整個螢幕 |
| `--region L T R B` | – | 自訂擷取區域 |
| `--fps` | 60 | 擷取目標 FPS |
| `--model` | yolo11n.pt | `yolo11s.pt` 更準但較慢 |
| `--imgsz` | 640 | 模型輸入尺寸，調小（如 480、416）更快 |
| `--conf` | 0.4 | 信心門檻 |
| `--monitor` | 0 | 擷取哪一個螢幕 |
| `--show-scale` | 0.6 | 預覽視窗縮放 |
| `--no-show` | – | 不開視窗，只印 FPS |

## 跑不到 60 FPS？

1. 確認啟動時印出的是 `CUDA - 你的顯卡`，不是 CPU。
2. 用較小的擷取區域（預設中央 640×640 就是為了速度）。
3. 調小 `--imgsz`，例如 `--imgsz 416`。
4. 進一步加速：把模型轉成 TensorRT（RTX 50 系列需要 TensorRT 10.8 以上）
   ```bash
   yolo export model=yolo11n.pt format=engine half=True imgsz=640
   python person_detector.py --model yolo11n.engine
   ```

## 注意

預覽視窗如果出現在擷取區域內，會出現「畫面套畫面」的無限迴圈。程式預設把視窗放在左上角並縮小；若仍重疊，可以把視窗拖到別的螢幕，或改用 `--region` 避開。

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
| `--size` | 0 | `0` = 整個螢幕；設 N 則只擷取中央 N×N 區域（較快） |
| `--region L T R B` | – | 自訂擷取區域 |
| `--fps` | 60 | 擷取目標 FPS |
| `--model` | yolo11n.pt | `yolo11s.pt` 更準但較慢 |
| `--imgsz` | 640 | 模型輸入尺寸。調小更快；全螢幕下遠處小人物抓不到可調大（960、1280） |
| `--conf` | 0.4 | 信心門檻 |
| `--monitor` | 0 | 擷取哪一個螢幕 |
| `--show-scale` | 0.5 | 預覽視窗縮放 |
| `--no-show` | – | 不開視窗，只印 FPS |
| `--cpu-preprocess` | – | 改用 ultralytics 原本的完整流程（較慢；快速流程有問題時才用） |

## 螢幕疊加模式（框直接畫在螢幕上）

不開預覽視窗，綠框直接疊在螢幕上，適合邊看影片邊偵測。疊加層透明、滑鼠點得穿，也不會被擷取進偵測畫面。

```bash
python person_overlay.py --model yolo11n.engine
```

要錄影（OBS、Windows 剪取工具等）時，加上 `--recordable`，錄影軟體才錄得到框：

```bash
python person_overlay.py --model yolo11n.engine --recordable
```

預設不加的話，框只有你眼睛看得到，錄影錄不到。

要結束：回到終端機按 `Ctrl + C`。參數跟 `person_detector.py` 相同（`--imgsz`、`--conf`、`--fps` 等）。

注意：Netflix 等有 DRM 保護的影片，在某些瀏覽器擷取到的會是黑畫面，這時偵測不到人；YouTube 一般沒有這個問題。

## 抓更遠、更小的人

4K 畫面預設被縮到 640 寬才偵測（縮小 6 倍），遠處的人只剩幾個像素就抓不到了。把輸入加大即可，建議用跟螢幕一樣 16:9 的長方形尺寸，不會浪費運算在補黑邊上：

```bash
yolo export model=yolo11n.pt format=engine half=True imgsz=[736,1280]
python person_overlay.py --model yolo11n.engine --imgsz 736 1280 --recordable
```

- `.engine` 的尺寸在轉換時就固定，`--imgsz` 必須跟轉換時一樣。
- 還想更準：把 `yolo11n.pt` 換成 `yolo11s.pt`（轉出來檔名是 `yolo11s.engine`）。
- 遠處的人分數通常比較低，可以再加 `--conf 0.25`。

## 沿著人的輪廓描邊

換成切割模型（檔名有 `-seg`）就會自動改成描輪廓，其他參數都一樣：

```bash
yolo export model=yolo11s-seg.pt format=engine half=True imgsz=[544,960]
python person_overlay.py --model yolo11s-seg.engine --imgsz 544 960 --recordable --hold 12 --fps 60
```

想換回方框，就把 `--model` 換回 `yolo11s.engine`。

## 第一人稱影片：不框自己的手

預設會自動忽略「碰到畫面最底部、而且從畫面下半部才開始」的框，也就是拍攝者自己露出來的手和手臂。站在前面的人通常從畫面中上方開始，不會被誤刪。

- 手還是被框到：把 `--self-top` 調小，例如 `--self-top 0.35`
- 前面的人被誤刪：把 `--self-top` 調大，例如 `--self-top 0.65`
- 不是第一人稱、不想過濾：加 `--keep-self`

## 忽略區域（遊戲介面等誤判）

預設忽略左下角（寬 0~11%、高 65~100%），框的中心落在這裡就不顯示。數字是畫面比例（0~1）：

```bash
python person_overlay.py ... --ignore 0 0.7 0.5 1            # 自訂範圍
python person_overlay.py ... --ignore 0 0.78 0.45 1 --ignore 0.8 0 1 0.2   # 多個區域
python person_overlay.py ... --no-ignore                      # 不忽略
```

## 框太多、太雜

追蹤會自動：合併同一個人的重複框、隱藏只出現一瞬間的誤判、讓抓不到的框停住而不是到處飄。還嫌雜可以再加：

- `--min-height 0.08`：框的高度小於畫面 8% 就不顯示（過濾遠處的人）
- `--max-boxes 10`：最多只顯示 10 個最大（最近）的框
- `--conf 0.5`：提高門檻，只框很確定的人

## 快速運動時框閃爍

預設開啟追蹤：
- 已經在追的人，畫面糊掉、分數變低（>= `--track-low`，預設 0.2）也繼續跟著；新出現的人要 >= `--conf` 才顯示。
- 暫時抓不到時，框會照之前的移動方向繼續走，最多保留 `--hold` 幀（預設 8）。
- 框的位置會自動平滑：人幾乎沒動時用 `--steady`（預設 0.15，越小越穩），人移動很快時用 `--smooth`（預設 0.7，越大越跟手）。

調整方式：
- 還是會閃：`--hold 15`
- 框跟不上動作：`--smooth 0.9`
- 人沒動時框還在抖：`--steady 0.08`
- 人離開了框還留著太久：`--hold 4`
- 關閉追蹤：`--no-track`

## 跑不到 60 FPS？

1. 確認啟動時印出的是 `CUDA - 你的顯卡`，不是 CPU。
2. 改成只擷取部分區域，例如 `--size 640`。
3. 調小 `--imgsz`，例如 `--imgsz 416`。
4. 進一步加速：把模型轉成 TensorRT（RTX 50 系列需要 TensorRT 10.8 以上）
   ```bash
   yolo export model=yolo11n.pt format=engine half=True imgsz=640
   python person_detector.py --model yolo11n.engine
   ```

## 注意

全螢幕擷取時，預覽視窗本身也在螢幕上，正常會出現「畫面套畫面」。程式會用 Windows 的 `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)` 把預覽視窗排除在擷取之外（需要 Windows 10 2004 以上）：你看得到視窗，但偵測畫面裡不會有它。

如果終端機出現「排除擷取失敗」的警告，可以把預覽視窗拖到第二個螢幕，或加 `--no-show`。

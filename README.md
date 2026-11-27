# 即時人物偵測：完整使用說明

看影片時，即時把畫面裡的人**框起來**或**沿著輪廓描邊**，可以直接錄影做成 YouTube 影片。
另外附一套「用自己的畫面訓練模型」的流程，讓模型在你常看的影片類型上更準。

- 系統：**Windows 10 / 11**
- 顯示卡：**NVIDIA**（以下以 RTX 5060 為例）
- 用途：真實影片（演唱會、樂團、車聚、街拍、運動、第一人稱影片等）。**不支援、也不要用在線上遊戲裡。**

---

## 目錄

1. [第一次安裝](#1-第一次安裝)
2. [下載與更新程式](#2-下載與更新程式)
3. [轉換模型（TensorRT）](#3-轉換模型tensorrt)
4. [開始使用](#4-開始使用)
5. [依照狀況調整](#5-依照狀況調整)
6. [所有參數一覽](#6-所有參數一覽)
7. [疑難排解](#7-疑難排解)
8. [用自己的畫面訓練模型（含 Gemini 檢查）](#8-用自己的畫面訓練模型含-gemini-檢查)
9. [注意事項](#9-注意事項)

---

## 1. 第一次安裝

**(1) 安裝 Python 3.12**

到 https://www.python.org/downloads/ 下載安裝。安裝畫面最下面要**勾選「Add python.exe to PATH」**。

**(2) 打開程式資料夾的黑色視窗**

打開放程式的資料夾 → 點上方網址列 → 輸入 `cmd` → Enter。
以下所有指令都在這個黑色視窗裡執行（滑鼠右鍵貼上）。

**(3) 安裝 PyTorch（顯示卡版）**

RTX 50 系列（5060/5070/5080/5090）**一定要用 cu128**，舊版會出現 `no kernel image is available`：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**(4) 安裝其他套件**

```bash
pip install -r requirements.txt
```

```bash
pip install tensorrt
```

**(5) 確認抓得到顯卡**

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

看到 `True NVIDIA GeForce RTX 5060` 就成功了。

---

## 2. 下載與更新程式

**第一次**：下載整包 zip 後解壓縮
https://github.com/juntinjc-cmyk/-/archive/refs/heads/claude/awesome-volta-p72gds.zip

**之後更新**：在黑色視窗執行下面幾行，會直接覆蓋成最新版（轉好的模型不會被刪）：

```bash
curl -L -o person_detector.py https://raw.githubusercontent.com/juntinjc-cmyk/-/claude/awesome-volta-p72gds/person_detector.py
```

```bash
curl -L -o person_overlay.py https://raw.githubusercontent.com/juntinjc-cmyk/-/claude/awesome-volta-p72gds/person_overlay.py
```

```bash
curl -L --create-dirs -o training/collect_frames.py https://raw.githubusercontent.com/juntinjc-cmyk/-/claude/awesome-volta-p72gds/training/collect_frames.py
```

```bash
curl -L -o training/auto_label.py https://raw.githubusercontent.com/juntinjc-cmyk/-/claude/awesome-volta-p72gds/training/auto_label.py
```

```bash
curl -L -o training/review_labels.py https://raw.githubusercontent.com/juntinjc-cmyk/-/claude/awesome-volta-p72gds/training/review_labels.py
```

```bash
curl -L -o training/train.py https://raw.githubusercontent.com/juntinjc-cmyk/-/claude/awesome-volta-p72gds/training/train.py
```

> GitHub 的連結有約 5 分鐘快取。剛更新完如果出現 `cannot import name ...`，代表拿到舊檔，等幾分鐘再下載一次。

---

## 3. 轉換模型（TensorRT）

TensorRT 會讓模型在你的顯卡上快很多。每個模型**只要轉一次**，每次要等幾分鐘，最後出現 `export success` 就完成。

| 想要的效果 | 轉換指令 | 產生的檔案 |
|---|---|---|
| 方框（推薦） | `yolo export model=yolo11s.pt format=engine half=True imgsz=[544,960]` | `yolo11s.engine` |
| 沿著人描輪廓 | `yolo export model=yolo11s-seg.pt format=engine half=True imgsz=[544,960]` | `yolo11s-seg.engine` |
| 方框，最快 | `yolo export model=yolo11n.pt format=engine half=True imgsz=[544,960]` | `yolo11n.engine` |
| 輪廓，最快 | `yolo export model=yolo11n-seg.pt format=engine half=True imgsz=[544,960]` | `yolo11n-seg.engine` |

- `n` 最快、`s` 比較準。用 `s` 的速度在 5060 上已經綽綽有餘。
- `.engine` 的尺寸在轉換時就固定了。執行時的 `--imgsz` **一定要跟轉換時一樣**（上表都是 `544 960`）。
- 想抓更遠的人，可以改成 `imgsz=[736,1280]`，執行時也要用 `--imgsz 736 1280`。比較慢但看得更遠。

---

## 4. 開始使用

### 推薦指令

**方框：**

```bash
python person_overlay.py --model yolo11s.engine --imgsz 544 960 --recordable --hold 12 --fps 60
```

**沿著人描輪廓：**

```bash
python person_overlay.py --model yolo11s-seg.engine --imgsz 544 960 --recordable --hold 12 --fps 60
```

- 影片請**全螢幕播放**。
- 螢幕左上角會出現綠色的 `FPS ... | persons ...`。
- **結束**：回到黑色視窗按 `Ctrl + C`。

### 啟動時應該看到

```
[資訊] 畫法: 方框（或：沿著人的輪廓描邊）
[資訊] 偵測流程: 快速（GPU 前處理 + 精簡後處理）
[資訊] 管線（擷取和推論同時進行）: 開
[資訊] 可錄影模式：錄影軟體錄得到框
```

接著每秒印一行：

```
FPS 60.0 | 擷取等待 4.0 ms | 偵測 6.0 ms（前處理 2.5 / 推論 3.5 / 後處理 0.3） | 人數 3
```

### 錄影

- 加了 `--recordable`，OBS、Windows 剪取工具都**錄得到框**。
- 不加的話，框**只有你看得到**，錄影錄不到。
- 有加 `--recordable` 時，偵測程式也會看到自己畫的框，偶爾可能造成一點誤判。

### 預覽視窗版（測試用）

不疊加在螢幕上，而是開一個視窗顯示偵測結果，並印出更詳細的速度資訊：

```bash
python person_detector.py --model yolo11s.engine --imgsz 544 960
```

按 `q` 或 `Esc` 結束。加 `--no-show` 則不開視窗，只印數字。

---

## 5. 依照狀況調整

把下面的參數加在指令最後面。

### 第一人稱影片：不要框到自己的手（預設開啟）

碰到畫面最底部、又從畫面下半部才開始的框，會被當成拍攝者自己的手。

| 狀況 | 加上 |
|---|---|
| 手還是被框到 | `--self-top 0.35`（越小過濾越多） |
| 前面的人被誤刪 | `--self-top 0.65` |
| 不是第一人稱，不想過濾 | `--keep-self` |

### 忽略某一塊畫面（預設：左下角）

框的中心落在這個範圍就不顯示。四個數字是**左、上、右、下**，用畫面比例 0～1 表示。
預設忽略左下角 `0 0.65 0.11 1`。

| 狀況 | 加上 |
|---|---|
| 自訂範圍 | `--ignore 0 0.7 0.5 1` |
| 好幾塊 | `--ignore 0 0.65 0.11 1 --ignore 0.8 0 1 0.2` |
| 完全不忽略 | `--no-ignore` |

### 快速動作時框會閃、抖、跟不上

追蹤功能預設開啟。

| 狀況 | 加上 |
|---|---|
| 框還是會閃、一下有一下沒有 | `--hold 15` |
| 框跟不上動作、慢半拍 | `--smooth 0.9` |
| 人沒動，框卻在抖 | `--steady 0.08` |
| 人離開了，框還留太久 | `--hold 4` |
| 關掉追蹤 | `--no-track` |

### 畫面太雜（人很多）

| 狀況 | 加上 |
|---|---|
| 遠處的小人不要框 | `--min-height 0.08`（高度不到畫面 8% 的不顯示） |
| 只框最近的幾個人 | `--max-boxes 10` |
| 只框很確定的人 | `--conf 0.5` |

### 太暗、太亮、模糊，抓不到人

| 狀況 | 加上 |
|---|---|
| 讓模型願意框不太確定的人 | `--conf 0.25`（可能多一些誤判） |
| 換更準的模型 | 改用 `yolo11s` 或用[第 8 節](#8-用自己的畫面訓練模型含-gemini-檢查)訓練 |

### 速度

| 狀況 | 做法 |
|---|---|
| 想讓影片播放更順 | `--fps 60`（螢幕是 60Hz 時，更高也看不出差別，只會搶顯卡） |
| 螢幕是 120Hz 以上 | `--fps 120` |
| FPS 不夠 | 換 `n` 模型，或用比較小的尺寸重新轉換 |

---

## 6. 所有參數一覽

`person_overlay.py` 和 `person_detector.py` 共用：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--model` | yolo11n.pt | 模型檔。檔名有 `-seg` 會自動改成描輪廓 |
| `--imgsz` | 640 | 模型輸入尺寸。一個數字（正方形）或「高 寬」兩個數字；要跟 `.engine` 轉換時一樣 |
| `--fps` | 60 | 擷取目標 FPS |
| `--conf` | 0.4 | 信心門檻（新出現的人） |
| `--size` | 0 | 0 = 整個螢幕；N = 只擷取螢幕中央 N×N |
| `--region L T R B` | – | 自訂擷取區域（像素） |
| `--monitor` | 0 | 擷取哪一個螢幕 |
| `--device` | 自動 | `0` = 第一張顯卡，`cpu` = 不用顯卡 |
| `--no-track` | – | 關閉追蹤 |
| `--hold` | 8 | 抓不到人時，框最多保留幾幀 |
| `--track-low` | 0.2 | 已經在追的人，分數掉到多低還繼續跟 |
| `--smooth` | 0.7 | 人移動很快時，框的跟手程度（越大越跟手） |
| `--steady` | 0.15 | 人幾乎沒動時，框的跟手程度（越小越穩） |
| `--min-height` | 0 | 框的高度小於畫面這個比例就不顯示 |
| `--max-boxes` | 0 | 最多顯示幾個框（0 = 不限） |
| `--ignore X1 Y1 X2 Y2` | 左下角 | 忽略區域（畫面比例），可以寫多次 |
| `--no-ignore` | – | 不使用忽略區域 |
| `--keep-self` | – | 不過濾第一人稱的手 |
| `--self-top` | 0.5 | 過濾手的靈敏度 |
| `--cpu-preprocess` | – | 改用 ultralytics 原本的流程（較慢，快速流程出問題時才用） |

只有 `person_overlay.py`：

| 參數 | 說明 |
|---|---|
| `--recordable` | 錄影軟體錄得到框 |
| `--no-pipeline` | 關閉管線（擷取和推論改成輪流做；管線出問題時才用） |

只有 `person_detector.py`：

| 參數 | 說明 |
|---|---|
| `--show-scale` | 預覽視窗縮放比例（預設 0.5） |
| `--no-show` | 不開預覽視窗，只印數字 |

---

## 7. 疑難排解

| 狀況 | 解法 |
|---|---|
| `no kernel image is available for execution on the device` | PyTorch 版本太舊，重新用 cu128 安裝（[第 1 節](#1-第一次安裝)） |
| 啟動時顯示 `CPU`，或 FPS 很低 | PyTorch 沒裝到顯卡版，重做第 1 節第 (3)(5) 步 |
| `cannot import name ...` | 兩個程式檔版本不一致，重新下載兩個檔案（等 5 分鐘快取過期） |
| 找不到 `.engine` 檔 | 還沒轉換，或檔名打錯（[第 3 節](#3-轉換模型tensorrt)） |
| 框的位置偏掉 | `--imgsz` 跟轉換時不一樣 |
| 錄影裡沒有框 | 沒加 `--recordable` |
| 螢幕上有殘留的綠色短線 | 會在 0.5 秒內自動清掉；一直存在的話截圖回報 |
| 螢幕整個變黑 | 立刻在黑色視窗按 `Ctrl + C`；不行就用工作管理員結束 Python |
| `快速流程無法使用` 的警告 | 會自動改用原本的流程，可以照常使用；把警告內容回報 |
| 疊加版出錯或框怪怪的 | 加 `--no-pipeline` 試試；還是不行再加 `--cpu-preprocess` |
| 有開其他很吃顯卡的程式 | 偵測會變慢，先關掉再測 |

---

## 8. 用自己的畫面訓練模型（含 Gemini 檢查）

讓模型在你常看的**真實影片**（暗的舞台、強光、動態模糊、人很擠）上更準。
**不要用遊戲畫面訓練。**

整個流程：

```
收集畫面 → 自動標註 →（可選）Gemini 檢查 →（可選）手動修正 → 訓練 → 比較分數 → 轉檔
```

### 步驟 1：收集畫面（至少 500～1000 張）

從影片檔抽畫面（畫質最好，推薦），每 2 秒一張：

```bash
python training/collect_frames.py --video 影片1.mp4 影片2.mp4 --every 2
```

或是播放影片時自動截圖，每 3 秒一張、截 60 分鐘（影片要全螢幕）：

```bash
python training/collect_frames.py --screen --every 3 --minutes 60
```

- 圖片存在 `dataset\raw\`。很像的畫面和黑畫面會自動跳過。
- 多收集模型**現在做不好**的畫面，效果最好。
- 隨時按 `Ctrl + C` 停止。

### 步驟 2：自動標註

```bash
python training/auto_label.py
```

- 用最大的模型 `yolo11x-seg`（第一次會自動下載）幫每張圖描出人的輪廓。
- 1000 張大約 10～20 分鐘。
- 自動留 1 成的圖當「考試用」，不拿去訓練。
- 重新執行會重做標註，之前 Gemini 檢查的結果也會清掉。

### 步驟 3（可選）：用 Gemini 檢查標註

Gemini 會幫你：
- **刪掉不是真人的標註**：海報、雕像、人形模特兒、畫面裡電視上的人、圖示等。
- **列出可能有漏標的圖片**，存在 `dataset\review_missed.txt`。

**(1) 安裝套件**

```bash
pip install google-genai
```

**(2) 設定金鑰**

到 Google AI Studio 取得金鑰。下面這行只對目前這個黑色視窗有效，關掉視窗要重設：

```bash
set GEMINI_API_KEY=你的金鑰
```

**(3) 先試跑 20 張**（只檢查、不修改），確認 Gemini 判斷得準：

```bash
python training/review_labels.py --limit 20 --dry-run
```

**(4) 正式執行**

```bash
python training/review_labels.py
```

- 自動控制速度，預設每分鐘 8 次請求。被限速時可以改成 `--rpm 5`。
- **進度會存檔**。今天的免費額度用完會自動停下，明天再執行同一行就會繼續。
- 修改前會先把原本的標註備份到 `dataset\labels_backup\`。
- 出現「模型名稱不存在」時，加 `--model gemini-2.5-flash`，或換成 AI Studio 裡看到的模型名稱。
- **只用一個帳號、一把金鑰**。用多個帳號輪流規避額度限制，違反 Google 的使用條款。
- 圖片會上傳給 Google，免費方案的資料可能被 Google 拿來改進產品。

### 步驟 4（可選）：手動修正

用免費的標註工具（例如 CVAT）打開 `dataset\`，補上 `review_missed.txt` 裡列出的漏標。這一步對準確度幫助最大。

### 步驟 5：訓練

```bash
python training/train.py
```

它會自動依序完成：
1. 幫**原本的模型**考試，記下分數。
2. 用你的顯卡訓練，1000 張大約 1～2 小時，顯卡會跑滿。
3. 幫**訓練後的模型**考試，印出分數變化。
4. **分數有變好才轉檔**，產生 `yolo11s-seg-custom.engine`；沒變好就不轉，不會動到你現在的模型。

中途停掉的話，可以接著練：

```bash
python training/train.py --resume
```

### 步驟 6：使用訓練好的模型

```bash
python person_overlay.py --model yolo11s-seg-custom.engine --imgsz 544 960 --recordable --hold 12 --fps 60
```

> 考試用的圖片也是大模型標的，所以分數代表「小模型有多接近大模型」。
> 有手動修正過考試用的圖片，分數才會比較接近真正的準確度。

---

## 9. 注意事項

- **只用在影片上**。不要在線上遊戲裡使用，會違反遊戲規則，也可能被封號。
- **版權**：上傳 YouTube 時，用自己拍的影片或允許使用的素材。Netflix 或別人的影片很容易被檢舉下架。
- **隱私**：車聚、演唱會影片裡的路人入鏡後上傳，可能涉及肖像權。
- **訓練資料**：`dataset\` 裡的截圖只留在自己電腦上，不要上傳或公開。這個資料夾、模型檔和訓練結果都不會被傳到 GitHub。
- **Gemini 金鑰**：不要貼給別人，也不要寫進程式裡。

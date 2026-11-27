# 專案交接：即時人物偵測疊加

這份檔案是給在使用者電腦上執行的 Claude Code 看的，讓新的對話能接續之前在雲端完成的工作。
使用者看得懂中文，這份也可以直接給使用者看。

## 使用者與溝通方式

- **一律用繁體中文回覆**（使用者看到英文會要求改回中文）。
- 使用者不是工程師：說明要白話，少用術語；必要的術語第一次出現時解釋一下。
- 每一步都給「可以直接執行」的指令，一次一個步驟。
- 改完程式後，請使用者實際跑跑看，並請他貼出終端機印的數字（FPS、各階段毫秒數）或截圖來確認。
- 現在可以直接在這台電腦上執行指令、讀取輸出，**能自己跑的就自己跑**，不要叫使用者複製貼上。
  需要看螢幕畫面的（框有沒有對準、影片播放順不順）才請使用者確認。

## 專案目標

看影片時（YouTube、Netflix、自己拍的影片：演唱會、樂團、車聚、運動、第一人稱影片），
**即時**把畫面中的人框起來，或沿著輪廓描邊，並且能錄影做成 YouTube 影片。
使用者明確要的是**即時**，不是離線處理影片檔。

## 界線（之前已經跟使用者講好，請維持）

- **不做任何跟線上遊戲有關的功能**：不在遊戲執行時使用或優化、不用遊戲畫面訓練模型、
  不做任何瞄準或滑鼠鍵盤自動化。理由：這等於遊戲透視和自動瞄準外掛，違反遊戲規則、可能被封號。
  使用者曾提出用遊戲畫面訓練，已經婉拒；看 YouTube 上的遊戲影片可以，但不為遊戲做特別調整。
- **Gemini API 只用一個帳號、一把金鑰**，遵守免費額度。使用者說過有很多帳號，
  已說明「多帳號輪流規避額度」違反 Google 條款，不提供。
- 金鑰只從環境變數 `GEMINI_API_KEY` 讀取，不要寫進任何檔案，也不要叫使用者貼在對話裡。
- 訓練用的截圖（`dataset/`）只留在本機，不要上傳。

## 硬體與環境

- Windows，NVIDIA **RTX 5060**（8 GB，Blackwell 架構），另有 AMD 內顯（不用它）。桌上型電腦。
- 螢幕 **3840×2160（4K）**，Windows 縮放比例未知；疊加視窗已經處理了縮放換算，框有對準。
- 螢幕更新率不確定（推測 60 Hz），建議用 `--fps 60`。
- 記憶體約 31 GB，常常用到 87%。
- Python + PyTorch 2.11.0+cu128（RTX 50 系列**一定要用 cu128**）、ultralytics（新版：`half` 參數已改名 `quantize`）、TensorRT、dxcam、PySide6、OpenCV。
- 專案資料夾：`C:\Users\junti\Desktop\--claude-awesome-volta-p72gds`
  （從 GitHub 的 zip 解壓縮而來，**不是 git clone**，之前都用 curl 下載單一檔案更新）。
- GitHub：`juntinjc-cmyk/-`，分支 `claude/awesome-volta-p72gds`。
- 使用者已轉好的 TensorRT 模型（`.engine`，尺寸都是 **544×960**，執行時要 `--imgsz 544 960`）：
  `yolo11n.engine`、`yolo11s.engine`、`yolo11s-seg.engine`。
  執行前先用 `dir *.engine` 確認實際有哪些、轉換尺寸是否一致。

## 檔案

- `person_detector.py`：核心偵測（`Detector` 類別、追蹤、過濾），也可單獨開預覽視窗測試。
- `person_overlay.py`：**主要使用的程式**。螢幕上的透明疊加層。
- `training/collect_frames.py`、`auto_label.py`、`review_labels.py`、`train.py`：微調模型流程。
- `README.md`：給使用者的完整中文說明（安裝、使用、參數、疑難排解、訓練）。

## 目前推薦的執行方式

```bash
# 方框
python person_overlay.py --model yolo11s.engine --imgsz 544 960 --recordable --hold 12 --fps 60
# 沿著輪廓描邊
python person_overlay.py --model yolo11s-seg.engine --imgsz 544 960 --recordable --hold 12 --fps 60
```

## 架構重點（每一項都是實測後才改成這樣，不要隨便改回去）

1. **擷取**：dxcam 擷取整個 4K 螢幕。快速流程用 BGRA，省掉 dxcam 的顏色轉換。
2. **快速偵測流程**（`Detector.prepare` / `detect_prepared`）：
   - CPU 用 cv2 縮小畫面 → pinned memory → 在獨立的 CUDA stream 上傳 → GPU 補灰邊。
   - **直接呼叫 TensorRT 模型**，略過 ultralytics 的 `predict`。原本的後處理要 8～10 ms。
   - 後處理只把 5 列資料（框＋「人」的分數）一次搬回 CPU，用 numpy 做 NMS，約 0.3 ms。
     在 GPU 上做很多小步驟反而比較慢，因為影片播放同時在用 GPU。
   - 切割模型：遮罩係數 × 原型遮罩在 GPU 算 → 每個人只處理自己框附近的範圍 → `cv2.findContours` 轉成輪廓。
   - `--cpu-preprocess` 會退回 ultralytics 原本的流程，用來排查問題。
3. **管線**（`person_overlay.py` 的 `Prefetcher`）：背景執行緒先擷取、縮小、上傳下一幀，主執行緒同時在推論這一幀。
   yolo11s 因此從 40～55 FPS 提升到 80～108 FPS。`--no-pipeline` 可以關閉。
4. **疊加視窗在獨立的 process**，用 multiprocessing Pipe 傳框。
   原本跟偵測放在同一個 process 裡，偵測時間會從 17 ms 變成 40 ms。
5. **透明用色鍵**（純黑 = 透明，`SetLayeredWindowAttributes` + `LWA_COLORKEY`），不用 `WA_TranslucentBackground`，
   後者每幀都要搬整個 4K 畫面。每幀只重畫框的邊線（輪廓模式改成重畫輪廓的外接範圍），
   另外每 0.5 秒全畫面重畫一次，清掉殘影。純黑會變透明，所以畫面上其他元素都不能用純黑色。
6. **排除擷取**：沒加 `--recordable` 時，疊加視窗用 `WDA_EXCLUDEFROMCAPTURE`，
   dxcam 和錄影軟體都看不到它。加了 `--recordable` 就錄得到框，但偵測也會看到自己畫的框。
7. **追蹤器**（`BoxTracker`）：
   - 雙門檻：新的人要 ≥ `--conf`（0.4），已在追的人只要 ≥ `--track-low`（0.2）。
   - 配對同時看 IoU 和中心點距離，快速移動也追得上。
   - 要連續出現 2 次才顯示；重疊超過 IoU 0.5 的重複框會合併。
   - 抓不到時沿原方向移動，速度很快衰減，最多保留 `--hold` 幀。
   - **自適應平滑**：幾乎沒動時用 `--steady`（0.15），動得快時用 `--smooth`（0.7）。
   - 輪廓不做平滑，直接用這一幀的偵測；抓不到時跟著框一起平移。
8. **過濾**：
   - 忽略區域：預設只有左下角 `0 0.65 0.11 1`，這是使用者自己指定的。
   - 第一人稱的手：框碰到畫面底部 97%，而且頂端低於 `--self-top`（0.5）。
   - `--min-height`、`--max-boxes`：減少畫面太雜。
9. 框和輪廓上**不畫文字**，這是使用者要求的。左上角保留 FPS 和人數。

## 實測數據（使用者電腦，Netflix/YouTube 4K 播放中）

- yolo11n 544×960：偵測約 6 ms，穩定 60 FPS（目標設 60 時）。
- yolo11s 544×960 加管線：80～108 FPS，推論 5～11 ms，波動是因為影片播放在搶 GPU。
- yolo11s-seg 544×960：近景約 60～65 FPS；樂團全景 19～22 人時 37～49 FPS。
  之後已改成「每人只處理自己的框」，**改完還沒實測**。
- 輪廓在快速動作時會落後約 1～2 幀，例如指揮揮手時。

## 還沒驗證／已知問題（優先處理）

1. **`training/` 四支程式都沒在 Windows 上實際跑過**，第一次執行很可能需要修小錯誤，例如路徑、套件版本、
   ultralytics 的 API 差異、Gemini 模型名稱（預設 `gemini-flash-latest`，不行就改 `gemini-2.5-flash`）。
   `train.py` 的 `score()` 用 80 類別的原始模型去考只有 1 類別的資料集，要確認 ultralytics 的 val 能正常跑。
2. 輪廓的「每人只處理自己的框」優化還沒實測：要確認輪廓有對準、全景時的 FPS。
3. 輪廓快速移動時的延遲。曾提議加「依速度提前預測位置」的選項，使用者還沒決定。
4. `--recordable` 會讓偵測看到自己畫的框，可能造成誤判。
5. 曾建議但還沒做的：
   - 雙擊就能執行的 `.bat` 啟動檔，或設定檔，讓使用者不用記長指令。
   - 只擷取影片播放的範圍，不擷取整個螢幕：比較準，也比較省資源。
   - 畫面沒變時跳過偵測，影片通常只有 24～30 FPS。
6. Netflix 在使用者電腦上**擷取得到**（不是黑畫面），已經確認過。

## 微調流程（使用者正準備開始）

```
collect_frames.py → auto_label.py →（可選）review_labels.py → 手動修正（CVAT，可選）→ train.py
```

`train.py` 會先幫原本的模型考試、再訓練、再幫訓練後的模型考試，**分數有變好才轉成** `yolo11s-seg-custom.engine`。
訓練資料只用真實影片，不用遊戲畫面。

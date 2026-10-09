# 色弱輔助即時校正系統

利用深度學習做即時（webcam）色弱/色盲輔助校正的專題。核心想法：用一個輕量化 U-Net
即時把畫面重新上色，讓色弱患者能分辨出原本會混淆的顏色，同時盡量維持畫面對一般人來說的自然度。

## 研究方向 (跟指導老師討論的結論)

現有文獻多半針對「單張圖片」做色弱校正，且分成兩派：
- **CNN 類**（Deep Correct、Daltonizer）：速度快但多半沒特別處理即時性。
- **GAN 類**（pix2pix / CycleGAN / BicycleGAN）：效果好但訓練不穩定，且文獻普遍
  提到會有顏色失真、不自然的問題，推論成本也較高。

2025 年的最新研究（如 Hue4U）才剛開始處理「即時 + 個人化」，代表這個方向仍有明顯的
研究空間，尤其是**一般 webcam 場景的即時影片校正**目前幾乎沒有文獻直接處理。

因此採取的策略：**自己搭建輕量 U-Net 架構**（而非直接套用他人的模型或走 GAN 路線），
把文獻中「模擬色弱視覺、比對並校正對比」的核心精神，用可學習的 loss function 實作，
之後再視情況與 GAN 類方法做效果/速度的比較實驗。

## 專案結構

```
project/
├── requirements.txt
├── data/
│   └── val2017/          # 訓練圖片 (COCO val2017, 5000張日常場景照片)
└── src/
    ├── cvd_simulation.py  # 色弱視覺模擬 (Machado et al. 2009 矩陣)
    ├── model.py           # 輕量 U-Net 架構
    ├── losses.py          # 自然度 + 結構(SSIM) + 可辨識度 loss
    ├── dataset.py          # 自監督訓練資料集
    ├── train.py            # 訓練腳本
    ├── eval_checkpoint.py  # 色弱可辨識度評估，結果累加到 results.csv
    ├── eval_metrics.py     # SSIM/MS-SSIM/CW-SSIM/LPIPS/PCDM 評估，結果累加到 metrics_summary.csv
    └── webcam_demo.py      # 即時 webcam demo
```

## 架構設計重點

- **U-Net + depthwise separable convolution**（MobileNet 風格輕量卷積），而非 GAN：
  這個任務是像素級顏色重映射，不需要生成新內容，U-Net 的 skip connection 能保留
  邊緣/物體輪廓，運算量也遠低於 GAN 的 generator+discriminator 架構，適合即時場景。
  目前模型約 **16 萬參數**。
- **殘差式輸出**：模型學的是「校正量」，校正後圖片 = 原圖 + 校正量。訓練起點就接近
  「不校正」，比較穩定，也降低顏色劇烈跑掉的風險。
- **色弱類型條件輸入**：色弱類型 (protanopia/deuteranopia/tritanopia) 編碼成額外
  channel 一起輸入，讓一個模型能服務三種類型，不用分開訓練。
- **自監督訓練**：不需要色弱標註資料。訓練資料只是一般彩色圖片，loss 用
  `cvd_simulation.py` 的生理模型去「模擬色弱患者看校正後圖片的感受」，並要求：
  1. 自然度（L1）：校正後圖片不能跟原圖差太多
  2. 結構（SSIM）：邊緣/物體輪廓要保留
  3. 可辨識度：色弱模擬後的對比，要盡量接近正常人看原圖的對比

## 已知問題與修正紀錄

- **NaN 訓練發散 bug**（已修正）：`cvd_simulation.py` 的 sRGB↔線性 RGB 互轉公式，
  `torch.where` 切換分支時，其中一個分支的梯度在輸入為 0 時是無限大，即使該分支
  沒被選中，PyTorch autograd 仍會算出 `0 × ∞ = NaN` 污染整個模型。修法是避免
  pow() 的輸入精確等於 0（clamp 到一個極小正數）。
- **Python 環境問題**：這台機器上 PATH 裡的 `python` 指向 MSYS64 的 Python（沒裝
  torch），實際要用的是 `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe`。
  已透過 `.vscode/settings.json` 設定 VSCode 預設直譯器解決。
- **色弱類型條件輸入形同虛設**（已修正，V6→V7）：`train.py` 算 distinguish/palette
  loss 時，不管模型當下被指定校正哪種色弱類型，永遠對 protanopia/deuteranopia/
  tritanopia 三種都算一次再平均。結果模型學到一套「不管輸入哪種類型都套用同一套
  通用校正」，實測（webcam demo）發現切到 tritanopia 模式時，校正後紅色反而被
  推向綠色。修法：`cvd_simulation.simulate_cvd_per_sample` + `losses.py` 的
  `cvd_type_idx` 參數，讓 loss 只用每張圖實際被指定的那一種類型去算。修正後
  （V7）用同一批圖測試，tritanopia 模式下純紅/純藍/純黃都能維持原本色調（V2:
  純紅被推向 (29,166,0) 綠色；V7: 純紅維持 (225,32,0) 紅色）。
  **注意**（已修正）：因為這個修正讓模型的校正變成真的跟色弱類型有關，
  `eval_checkpoint.py` 原本「不管條件輸入、三種類型都算再平均」的評分方式，
  對這種「有做條件式校正」的模型反而不公平（會低估其表現），已改成用每張圖
  實際被指定的 `cvd_type_idx` 去算 loss。`results.csv` 裡 V6/V7 那兩列是用
  舊評分方式算的，數字看起來比 V2~V5 差，是這個舊 bug 造成的假象，不代表
  V6/V7 實際更差——用修正後的評分方式重新測，V7 在三種色弱類型上都優於
  V2（distinguish loss 再降 15~59%、palette loss 再降 41~70%，見
  `_verify_v7_fair.py`），也優於完全不校正（58~86%）。
- **train/test 沒切分**（已修正）：`dataset.py`/`train.py`/`eval_metrics.py`/
  `eval_checkpoint.py` 原本都指向同一份 `data/val2017`，模型評估時用的是訓練時
  看過的圖，數字可能只是背答案的假象。已加上 `dataset.split_image_paths()`
  切出固定的 train(4500)/test(500)，`train.py` 只用 train、eval 腳本預設只用
  test。**但 checkpoints_v2~v7 都是切分機制加入前訓練的**，這些版本沒辦法用
  test split 得到乾淨數字。為了驗證「V7 是否只是背答案」，另外訓練了 V8
  （設定同 V7，只是只用 train split 訓練），在從沒看過的 test split 上比較
  （見 `_verify_v8_holdout.py`）：
  - **可辨識度（distinguish/palette loss）在三種色弱類型上都有生成**：V8 沒看過
    這 500 張圖，效果仍逼近甚至超過看過這些圖的 V2（例如 deuteranopia
    distinguish 改善 V8 49.2% vs V2 30.4%），證實這個核心效果是真的學到
    可泛化的校正策略，不是背答案。
  - **但保真度指標（SSIM/LPIPS/PCDM）在 tritanopia 上有明顯的 train/test 落差**：
    同一批 500 張 held-out 圖，V7（看過）tritanopia PCDM=6.86、LPIPS=0.064，
    V8（沒看過）PCDM=11.06、LPIPS=0.097，protanopia/deuteranopia 則沒有這種
    落差（V8 甚至持平或略贏）。代表 tritanopia 的顏色校正相對更吃訓練資料量/
    多樣性，泛化能力比另外兩種類型弱，是目前一個真實存在、值得在報告裡揭露
    的限制，之後可以考慮增加 tritanopia 相關的訓練資料或加強正則化。

## 目前主模型：V9

跟老師討論後（2026-10-08）決定**以 V9 為主模型**，V8 當對照組。之後新的評估、
demo 都預設用 `src/checkpoints_v9/model_epoch50.pt`。

- **V9 = V8 + 針對性混淆色對合成資料**（`--targeted_ratio 0.15`，三種色弱類型
  隨機混合，見 `synthetic_colors.ConfusionPairBlocks`）。
- **選 V9 的理由**：三種色弱類型的保真度最平均。V8 在 tritanopia 上明顯失真，
  V9 把它拉回跟另外兩種類型同一個水準（held-out 500 張，`metrics_summary.csv`）：

  | PCDM（越低越好） | protanopia | deuteranopia | tritanopia |
  |---|---|---|---|
  | V8 | 5.25 | 4.96 | **11.06** |
  | V9 | 7.80 | 7.52 | **7.14** |

- **代價（報告裡要揭露）**：V9 的 protan/deuter 保真度比 V8 差一些（PCDM 約
  5→7.5），可辨識度改善幅度也比 V8 小（例如 tritanopia distinguish 改善 V8
  70.1% → V9 64.3%，protanopia 67.8% → 62.7%，見 `_verify_v10_targeted.py`）。
  也就是 V9 用「三種類型校正力道都稍微收斂」換取「tritanopia 不會特別差」。
- V10（只補 tritanopia 的合成資料）的 tritan 可辨識度比 V8 還差，已排除，
  完整經過見 `_verify_v10_targeted.py` 開頭說明。

## tritanopia 的模擬可信度（Brettel 1997 重評）

訓練跟評估用的 Machado 2009 矩陣，原作者明說沒有真正建模 tritanopia，文獻
（DaltonLens 2021 對開源模擬法的評比）建議 tritan 改用 Brettel, Viénot & Mollon 1997。
所以在 `cvd_simulation.py` 加上 Brettel 1997 的 tritan 模擬（`set_tritan_model`，
預設仍是 Machado，訓練不受影響；數值已對 `daltonlens` 參考實作驗證），並用它
重新評估 held-out 500 張的 tritan 可辨識度（`_verify_brettel_tritan.py`）：

| tritan 可辨識度改善 | Machado（訓練用） | Brettel（較可信） |
|---|---|---|
| V8 distinguish / palette | +70.1% / +83.2% | +55.2% / +54.8% |
| V9 distinguish / palette | +64.4% / +73.1% | +55.0% / +54.2% |

- 換成較可信的模擬，**校正效果仍在**（約 +55%），代表模型學到的 tritan 校正是真的，
  不只是對某個特定模擬矩陣有效。
- 但改善幅度比用 Machado 量時小 15~30 個百分點：模型有一部分是在「迎合」訓練用的
  Machado 矩陣，用 Machado 報的 tritan 數字會高估效果。
- 在 Brettel 下 **V8 跟 V9 幾乎一樣**，V9 在 tritan 可辨識度上的「代價」只在 Machado
  下才看得到，更支持以 V9 為主模型。
- 只影響可辨識度：`eval_metrics.py` 的 SSIM/LPIPS/PCDM 是「原圖 vs 校正後」，
  不經過色弱模擬，換模擬模型不會改變。
- 兩種模擬對同一張原圖的 tritan 結果平均差 ΔE2000 = 3.25，屬於肉眼可察覺的落差。

## 使用方式

安裝套件：
```
pip install -r requirements.txt
```

訓練：
```
cd src
python train.py --data_dir "../data/val2017" --epochs 50 --batch_size 8 --targeted_ratio 0.15 --checkpoint_dir checkpoints_v9
```
（上面是主模型 V9 的設定；拿掉 `--targeted_ratio 0.15` 就是對照組 V8。）

影像品質指標評估（SSIM / MS-SSIM / CW-SSIM / LPIPS / PCDM，用來跟其他方法比較）：
```
cd src
python eval_metrics.py --checkpoint checkpoints_v9/model_epoch50.pt --tag v9
```
預設三種色弱類型都算；每張圖的結果在 `metrics_results/`（不進版控），各版本平均值累加到
`metrics_summary.csv`。這些指標量的是「校正後跟原圖有多像」（保真度），不是校正效果；
跟其他方法比較時，資料集、圖片大小(256x256)、色弱類型都要對齊。CW-SSIM 用標準的 7x7
局部視窗版（灰階），跟論文報的數值才有可比性。

即時 webcam demo（訓練完成後）：
```
cd src
python webcam_demo.py
```
預設載入主模型 `checkpoints_v9/model_epoch50.pt`；要看其他版本加 `--checkpoint`，
例如 `--checkpoint checkpoints_v8/model_epoch50.pt`。
鍵盤操作：`1`/`2`/`3` 切換色弱類型（protanopia/deuteranopia/tritanopia），`q` 離開。
畫面會並排顯示：原始畫面 / 校正前的色弱模擬 / 校正後的色弱模擬，方便直接比較效果。

## 訓練環境

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB)
- 訓練資料: COCO val2017，5000 張圖片，256x256 隨機裁切

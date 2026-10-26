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
│   ├── val2017/          # 訓練圖片 (COCO val2017, 5000張日常場景照片)
│   └── kodak/            # 跨資料集測試用 (Kodak 24 張，只推論不訓練)
└── src/
    ├── cvd_simulation.py  # 色弱視覺模擬 (Machado et al. 2009 矩陣)
    ├── model.py           # 輕量 U-Net 架構
    ├── losses.py          # 自然度 + 結構(SSIM) + 可辨識度 loss
    ├── dataset.py          # 自監督訓練資料集
    ├── train.py            # 訓練腳本
    ├── eval_checkpoint.py  # 色弱可辨識度評估，結果累加到 results.csv
    ├── eval_metrics.py     # SSIM/MS-SSIM/CW-SSIM/LPIPS/PCDM 評估，結果累加到 metrics_summary.csv
    ├── eval_literature_metrics.py  # 文獻指標 GCD/GCE/LCE/CD/FSIMc，結果累加到 literature_metrics_summary.csv
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
    的限制。後續處理見下方「目前主模型：V9」與「tritanopia（藍黃色弱）的限制」。

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

## tritanopia（藍黃色弱）的限制

老師 10/8 問到「藍黃是不是效果最差、有沒有文獻」。整理後的結論分三層：

### 1. 文獻沒有一致認定 tritan 校正效果最差

- 查不到任何文獻直接比較三種類型、並指出 tritan 校正最差。有分類型報結果的研究，
  結論也不一致：例如一篇用 LMS + ResNet + CycleGAN 的研究 [1]，10 位受試者的
  辨色正確率是 deutan 71.7%、tritan 59.7%、**protan 46.5% 反而最差**。
- 比較接近的說法只有質性描述：[2] 指出 tritan 用 color shifting 或 LMS 校正時，
  「may produce a new confusion」（可能產生新的混淆色）。
- 真正明確的是 **tritan 被研究得很少**：survey [3] 整理的校正方法幾乎都只針對紅綠
  色弱；Hue4U [4] 的受試者也以 protan/deutan 為主（兩者佔 CVD 超過 95%）。

所以報告不宜寫「文獻指出藍黃效果最差」，比較準確的說法是「tritan 研究少、
評估也較不可靠」。

### 2. 常用的 tritan 模擬模型本身不可靠

本專案訓練與評估都用 Machado 2009 [5] 的模擬矩陣，但原作者明說沒有真正建模
tritanopia，tritan 矩陣只是近似 [6]；Viénot 1999 的方法也不適用 tritan。對開源模擬法
的評比 [7] 結論是 Brettel, Viénot & Mollon 1997 [8] 是 tritan「basically only valid
choice」。因為我們的 loss 和可辨識度評估都依賴模擬，**tritan 的數字可信度本來就
比 protan/deutan 低**。

### 3. 用較可信的 Brettel 1997 重新評估（換尺，不換模型）

在 `cvd_simulation.py` 加上 Brettel 1997 的 tritan 模擬（`set_tritan_model`，預設仍是
Machado，訓練不受影響；數值已對 `daltonlens` 參考實作驗證）。**模型沒有重訓**，
只是把「評分時模擬藍黃色盲看到的畫面」換成 Brettel，重新量 held-out 500 張的
tritan 可辨識度（`_verify_brettel_tritan.py`）：

| tritan 可辨識度改善 | 用 Machado 量（訓練用） | 用 Brettel 量（較可信） |
|---|---|---|
| V8 distinguish / palette | +70.1% / +83.2% | +55.2% / +54.8% |
| V9 distinguish / palette | +64.4% / +73.1% | +55.0% / +54.2% |

- **校正效果是真的**：換一個模型訓練時沒見過的模擬來量，仍改善約 55%，
  不是只對 Machado 矩陣有效的假象。
- **但用 Machado 報的 tritan 數字會高估**：換成 Brettel 後少了 15~30 個百分點，
  模型有一部分是在「迎合」訓練用的模擬矩陣。報告裡 tritan 的結果兩種都列。
- **Brettel 下 V8 與 V9 幾乎一樣**：V9 在 tritan 可辨識度上的代價只在 Machado 下
  看得到，更支持以 V9 為主模型。
- 兩種模擬對同一張原圖的 tritan 結果平均差 ΔE2000 = 3.25，屬肉眼可察覺的落差，
  選哪個模擬模型不是小細節。

### 仍存在的限制與後續

- 換模擬只影響可辨識度。保真度指標（SSIM/LPIPS/PCDM）比的是「原圖 vs 校正後」，
  不經過模擬，所以 V8 的 tritan PCDM 偏高（11.06）無法用模擬不準來解釋。
- Brettel 本身也只是「較可信的近似」，沒有針對 tritan 受試者驗證過的模擬模型；
  要真正確認效果仍需 tritan 受試者實測，但 tritan 盛行率低，受試者難找。
- 可能的改善：用 Brettel 當 tritan 的訓練 loss 重訓（V11），讓模型針對較準的模擬
  校正，而不是迎合 Machado。尚未執行。

### 參考文獻

1. A. P. Adyani, *Koreksi Warna Pada Citra untuk Penderita Buta Warna Menggunakan
   Representasi Warna LMS dan CNN*, 學士論文, UPN Veteran Jawa Timur.
   https://repository.upnjatim.ac.id/39265/
2. *Smartphone Based Image Color Correction for Color Blindness*, Int. J. Interactive
   Mobile Technologies (iJIM), 2018. https://online-journals.org/index.php/i-jim/article/view/8160
3. Z. Zhu, X. Mao, "Image recoloring for color vision deficiency compensation: a survey,"
   *The Visual Computer*, 2021. https://doi.org/10.1007/s00371-021-02240-0
4. J. Qin et al., "Hue4U: Real-Time Personalized Color Correction in Augmented Reality,"
   arXiv:2509.06776, 2025.
5. G. M. Machado, M. M. Oliveira, L. A. F. Fernandes, "A Physiologically-based Model for
   Simulation of Color Vision Deficiency," *IEEE TVCG* 15(6), 2009.
6. colour-science, `colour.blindness.machado2009` 文件說明.
   https://colour.readthedocs.io/en/latest/_modules/colour/blindness/machado2009.html
7. DaltonLens, "Review of Open Source Color Blindness Simulations," 2021.
   https://daltonlens.org/opensource-cvd-simulation/
8. H. Brettel, F. Viénot, J. D. Mollon, "Computerized simulation of color appearance for
   dichromats," *JOSA A* 14(10), 1997.

## 影像量化：混淆色對與石原式檢查圖

回應老師 10/8 第 4 點（用影像量化改善效果），見 `_quantify_cvd_correction.py`。
在「色弱者會混淆的顏色對」上，量校正前 / 後在色弱模擬下兩色的 CIEDE2000 色差
（ΔE2000 約 1 為剛可察覺的差異，越大越分得出來）：

- **混淆色對色塊**：左半顏色 A、右半顏色 B。
- **石原式檢查圖**：圓點拼成的圓盤，數字的點用 A、背景的點用 B，每個點亮度隨機抖動
  （跟真的石原圖一樣，不能只靠亮度認出數字），量「數字 vs 背景」的平均色差。

混淆色對是**自動搜尋**的，不是手挑：在 13³ 的 RGB 網格上找「正常視覺 ΔE > 25、
色弱模擬後 ΔE < 3」的顏色對，每種類型挑 8 組。V9 訓練時用過
`synthetic_colors._CONFUSION_PAIRS` 的混淆色塊，所以跟那些訓練色 ΔE < 15 的顏色
全部排除，避免考到練過的題目。

| 色弱模擬下 ΔE2000（8 組平均） | 校正前 | V9 | V8 | 自然度代價 V9 / V8 |
|---|---|---|---|---|
| protan 色塊 | 1.31 | 48.57 | 51.58 | 28.0 / 30.0 |
| protan 檢查圖 | 1.26 | 40.65 | 41.90 | 22.3 / 22.7 |
| deutan 色塊 | 1.89 | 35.30 | 42.27 | 24.5 / 30.0 |
| deutan 檢查圖 | 1.75 | 31.32 | 29.99 | 20.1 / 19.9 |
| tritan（Brettel）色塊 | 2.13 | 24.19 | 33.05 | 16.7 / 31.0 |
| tritan（Brettel）檢查圖 | 2.03 | 20.68 | 26.31 | 13.4 / 21.0 |

（自然度代價 = 校正後顏色跟原色的 ΔE2000，正常視覺下，越小越自然。每組明細在
`confusion_pairs_results.csv`，長條圖 `quantify_pairs.png`、檢查圖範例
`quantify_plates.png` 由腳本重新產生。）

- **三種類型都從「分不出來」變成「明顯分得出來」**：校正前數字跟背景在模擬下只差
  ΔE 1~2（看不到數字），校正後差 20~49。
- **tritan 改善幅度最小**（V9 檢查圖 20.7 vs protan 40.7），跟可辨識度指標的趨勢一致。
- **V9 vs V8**：V8 大多分得更開（deutan 檢查圖例外，兩者相近），但顏色也改得更多；V9 的自然度代價明顯較小（tritan
  色塊 16.7 vs 31.0），跟「V9 校正較收斂、較自然」的取捨一致。
- **代價不小**：混淆色本身要被改 ΔE 13~31 才分得開。這是預期內的（不改色就不可能
  分開），但也代表這類高飽和混淆色在校正後，一般視覺看起來會明顯變色。實際照片的
  平均色差（PCDM）只有 5~8，因為大部分像素不是混淆色。
- **tritan（Machado）沒有結果**：用 Machado 的 tritan 模擬，排除訓練色後找不到任何
  符合條件的混淆色對。Machado 的 tritan 模擬太弱，強混淆色幾乎都集中在訓練用過的
  那幾組，又一次說明它不適合用來評估 tritan（見上一節）。

## 跨資料集泛化：Kodak 24 張

回應老師 10/8 第 1 點（交叉比對泛化能力）。V8 的 held-out 驗證只是 COCO 內部切分，
test 跟 train 分布很接近；這裡把在 COCO 訓練的 V9（主模型）/ V8（對照）**直接拿到
Kodak 24 張上推論，不重訓**。Kodak 是不同相機、不同題材的高畫質照片（需自行下載到
`data/kodak/`，不進版控，來源 `https://r0k.us/graphics/kodak/`）。可辨識度見
`_eval_kodak.py`，保真度用 `eval_metrics.py --data_dir ../data/kodak --split all`
（tag `kodak_v9` / `kodak_v8`，結果在 `metrics_summary.csv`）。兩邊都直接 resize 成
256x256，前處理一致。

**可辨識度：相對不校正的改善 %（distinguish / palette，越高越好）**

| 類型 | V9 COCO test | V9 Kodak | V8 COCO test | V8 Kodak |
|---|---|---|---|---|
| protan | 61.6 / 73.3 | 59.1 / 68.7 | 66.4 / 78.3 | 63.6 / 72.4 |
| deutan | 43.0 / 55.7 | 44.8 / 54.4 | 48.4 / 64.5 | 49.0 / 60.8 |
| tritan（Machado） | 63.4 / 73.5 | 65.5 / 73.4 | 69.1 / 83.6 | 66.4 / 76.1 |
| tritan（Brettel） | 54.1 / 55.2 | 55.1 / 48.1 | 54.3 / 55.6 | 57.0 / 49.2 |

**保真度（原圖 vs 校正後，PCDM = CIEDE2000 平均，越低越自然；LPIPS 越低越好）**

| 類型 | V9 COCO test | V9 Kodak | V8 COCO test | V8 Kodak |
|---|---|---|---|---|
| protan PCDM / LPIPS | 7.80 / 0.069 | 9.05 / 0.096 | 5.25 / 0.047 | 6.03 / 0.068 |
| deutan PCDM / LPIPS | 7.52 / 0.060 | 9.42 / 0.091 | 4.96 / 0.036 | 7.33 / 0.065 |
| tritan PCDM / LPIPS | 7.14 / 0.056 | 9.11 / 0.074 | 11.06 / 0.097 | 14.06 / 0.127 |

（SSIM / MS-SSIM / CW-SSIM 在兩個資料集都在 0.96~0.995，差異小，完整數字見 `metrics_summary.csv`。）

- **校正效果可以泛化到別的資料集**：可辨識度改善在 Kodak 上跟 COCO test 差不多，
  distinguish 差在 ±3 個百分點內（有幾格 Kodak 反而比較好）。掉得比較多的是 palette
  （大範圍色塊間的對比）：protan 掉 4.6~5.9、Brettel tritan 掉 6.4~7.1、V8 tritan（Machado）
  掉 7.5 個百分點，但改善幅度仍有約 50% 以上。
- **保真度在 Kodak 上全面變差一些**：PCDM 高 1~3、LPIPS 高 0.02~0.03，代表模型在
  Kodak 上改色改得比較多。這不是少數幾張圖拉高的（V9 tritan PCDM 中位數 COCO 6.5 →
  Kodak 9.4），也**不是**因為 Kodak 比較鮮豔（Hasler–Süsstrunk colorfulness：COCO
  40.6、Kodak 39.3，幾乎一樣）。確切原因還沒查清楚。
- **V9 vs V8 的取捨在 Kodak 上維持一樣**：V8 可辨識度略高、但 protan/deutan 改得更多；
  tritan 則是 V9 自然很多（PCDM 9.1 vs 14.1），而 Brettel 下兩者可辨識度相當，
  再次支持以 V9 為主模型。
- **限制**：Kodak 只有 24 張，統計誤差大（例如 V9 tritan PCDM 9.11 ± 2.56，平均值的
  標準誤約 0.5）；也還沒做多 seed 重訓，數字是單一模型的結果。

## 指標的文獻依據與文獻指標評估

回應老師 10/8 第 2 點（指標需要文獻驗證）。

### 1. 現有保真度指標的出處

| 指標 | 出處 | 在綜述中的角色 |
|---|---|---|
| SSIM | Wang et al. 2004 [S1] | 綜述列為自然度指標 |
| MS-SSIM | Wang, Simoncelli, Bovik 2003 [S2] | — |
| CW-SSIM | Sampat et al. 2009 [S3] | — |
| LPIPS | Zhang et al. 2018 [S4] | — |
| PCDM（CIEDE2000 平均） | Luo, Cui, Rigg 2001 [S5]；實作依 Sharma et al. 2005 [S6] | 綜述的 CD 指標同一類（CD 用 CIELAB 歐氏距離） |
| colorfulness（Kodak 段落用） | Hasler & Süsstrunk 2003 [S7] | — |

MS-SSIM、CW-SSIM、LPIPS 是通用的影像品質指標，CVD 校正文獻比較少用。綜述
（Zhu & Mao 2021，上一節參考文獻 3）第 3.2 節整理的 dichromacy 校正常用量化指標是：
對比用 GCD / GCE / LCE，自然度用 CD / SSIM / FSIMc。

### 2. 自創指標的問題，以及改用文獻指標

distinguish / palette 是我們自己定義的，**同時也是訓練 loss**，拿來評估等於
球員兼裁判。`eval_literature_metrics.py` 依綜述的公式實作了文獻指標（全部在
CIELAB 上算，對比指標都經過色弱模擬）：

- **GCD**（Ma et al. 2008 [S8]；Chen et al. 2011 [S9]）：色弱者看到的全域對比強度，
  越高越好。這裡報「補回 %」=（校正後 − 不校正）/（原圖正常視覺 − 不校正），
  也就是色弱損失掉的對比補回幾成。
- **GCE**（Huang et al. 2007 [S10]；Hassan 2019 [S11]）：顏色對之間的對比跟原圖
  不一致的程度，越低越好。概念對應我們的 palette。
- **LCE**（Machado & Oliveira 2010 [S12]；Zhu et al. 2019 [S13]；Wang et al. 2021 [S14]）：
  像素跟鄰居的局部對比跟原圖不一致的程度，越低越好。概念對應我們的 distinguish。
- **CD**、**FSIMc**（Zhang et al. 2011 [S15]，用 piq 套件）：自然度。

GCE / LCE 是「對比要跟原圖一致」，對比不夠或太強都會扣分；GCD 只看夠不夠強。
綜述沒規定的細節（GCE/GCD 每張圖抽 512 個像素、LCE 在 15x15 鄰域抽 8 個鄰居）是我們
自己選的，寫在腳本開頭。綜述表 3~6 是在他們自己的 6 張圖上算的，**數字不能直接對照**。

**結果（V9；括號是 V8；改善 % 都是相對不校正）**

| | GCD 補回 ↑ | GCE 改善 ↑ | LCE 改善 ↑ | CD ↓ | FSIMc ↑ |
|---|---|---|---|---|---|
| protan COCO test | 56.0% (47.4%) | +7.1% (+8.4%) | −19.2% (−7.9%) | 8.67 (6.44) | 0.987 (0.990) |
| protan Kodak | 36.3% (32.1%) | +35.1% (+34.9%) | +16.3% (+12.1%) | 9.46 (7.32) | 0.985 (0.987) |
| deutan COCO test | 51.6% (47.7%) | −14.1% (−14.2%) | −32.9% (−26.7%) | 8.23 (5.84) | 0.990 (0.994) |
| deutan Kodak | 36.3% (35.3%) | +22.9% (+25.3%) | −8.2% (−2.4%) | 9.55 (8.04) | 0.987 (0.990) |
| tritan Brettel COCO test | 5.2% (10.4%) | +4.7% (+9.0%) | −7.1% (−14.3%) | 8.24 (12.71) | 0.989 (0.981) |
| tritan Brettel Kodak | −7.7% (−7.1%) | −7.9% (−12.8%) | −22.6% (−30.4%) | 9.91 (14.58) | 0.988 (0.974) |

（tritan Machado 和逐項數字見 `literature_metrics_summary.csv`。）

**結論：用文獻指標看，校正效果比自創指標顯示的小很多。**

- **紅綠色弱（protan/deutan）有效，但只補回一部分**：GCD 顯示色弱者損失的全域對比
  補回 32~56%。這跟「影像量化」一節混淆色對的大幅改善一致，只是在一般照片上被稀釋了。
- **但對比的「位置」沒有對準原圖**：LCE 大多變差，GCE 時好時壞。拆開來看（診斷時
  另外算的，COCO test 96 張），不校正時誤差幾乎全是「對比不夠」（過強的部分 < 6%），
  校正後有 10~49% 的誤差來自「對比比原圖還強」。也就是模型把對比補回來了，但會在
  原本沒有對比的地方加出新的對比，補的位置也不完全對。
- **自創指標偏樂觀的原因是色彩空間跟尺度**：distinguish / palette 是在 sRGB 上算，
  而且主要的改善來自縮小 4 倍、16 倍後的粗尺度。同樣的「像素對距離誤差」，在 sRGB 上算
  （15x15 鄰域）protan V9 改善 +16.6%，換到 Lab 上就變成 −25.9%；在像素尺度
  （3x3 鄰域）兩種空間都變差。模型最佳化的是 sRGB 上的粗尺度對比，換成感知均勻的
  CIELAB 來看，效果就沒那麼好。
- **tritan 效果最弱**：GCD 只補回 5~30%（COCO），在 Kodak 上甚至略低於不校正。
  這跟前面「tritan 改善幅度最小」的結論一致，而且比自創指標顯示的更差。
- **自創指標跟文獻指標的方向一致，但只是中度相關**：把每張圖的數值放在一起看，
  distinguish~LCE 的 Spearman 為 0.34~0.74、palette~GCE 為 0.44~0.72。相關性大多來自
  「圖本身難不難」，不足以證明自創指標可以取代文獻指標。
- **後續建議**：報告的主要數字應改用 GCD / LCE / CD / FSIMc，自創指標只當輔助。
  如果要改善模型，訓練 loss 可以改在 CIELAB 上算、並加強像素尺度的權重（要重訓）。

### 參考文獻（指標）

- [S1] Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, "Image quality assessment: from
  error visibility to structural similarity," *IEEE TIP* 13(4), 600–612, 2004.
- [S2] Z. Wang, E. P. Simoncelli, A. C. Bovik, "Multiscale structural similarity for image
  quality assessment," *Asilomar Conf. Signals, Systems & Computers*, 2003.
- [S3] M. P. Sampat, Z. Wang, S. Gupta, A. C. Bovik, M. K. Markey, "Complex wavelet structural
  similarity: a new image similarity index," *IEEE TIP* 18(11), 2385–2401, 2009.
- [S4] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, "The unreasonable effectiveness
  of deep features as a perceptual metric," *CVPR*, 2018.
- [S5] M. R. Luo, G. Cui, B. Rigg, "The development of the CIE 2000 colour-difference formula:
  CIEDE2000," *Color Research & Application* 26(5), 340–350, 2001.
- [S6] G. Sharma, W. Wu, E. N. Dalal, "The CIEDE2000 color-difference formula: implementation
  notes, supplementary test data, and mathematical observations," *Color Research &
  Application* 30(1), 21–30, 2005.
- [S7] D. Hasler, S. Süsstrunk, "Measuring colourfulness in natural images," *Proc. SPIE*
  5007, Human Vision and Electronic Imaging VIII, 2003.
- [S8] Y. Ma, X. Gu, Y. Wang, "Color discrimination enhancement for dichromats using
  self-organizing color transformation," *Information Sciences* 179(6), 830–843, 2008.
- [S9] W. Chen, W. Chen, H. Bao, "An efficient direct volume rendering approach for
  dichromats," *IEEE TVCG* 17(12), 2144–2152, 2011.
- [S10] J.-B. Huang, Y.-C. Tseng, S.-I. Wu, S.-J. Wang, "Information preserving color
  transformation for protanopia and deuteranopia," *IEEE Signal Processing Letters* 14(10),
  711–714, 2007.
- [S11] M. Hassan, "Flexible color contrast enhancement method for red-green deficiency,"
  *Multidimensional Systems and Signal Processing* 30(4), 1975–1989, 2019.
- [S12] G. M. Machado, M. M. Oliveira, "Real-time temporal-coherent color contrast enhancement
  for dichromats," *Computer Graphics Forum* 29(3), 933–942, 2010.
- [S13] Z. Zhu, M. Toyoura, K. Go, I. Fujishiro, K. Kashiwagi, X. Mao, "Naturalness- and
  information-preserving image recoloring for red–green dichromats," *Signal Processing:
  Image Communication* 76, 68–80, 2019.
- [S14] X. Wang, Z. Zhu, X. Chen, K. Go, M. Toyoura, X. Mao, "Fast contrast and naturalness
  preserving image recolouring for dichromats," *Computers & Graphics* 98, 19–28, 2021.
- [S15] L. Zhang, L. Zhang, X. Mou, D. Zhang, "FSIM: a feature similarity index for image
  quality assessment," *IEEE TIP* 20(8), 2378–2386, 2011.

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

文獻上的 CVD 校正指標（GCD / GCE / LCE / CD / FSIMc，見「指標的文獻依據」一節；
COCO test 跟 Kodak 都算，Kodak 要先下載到 `data/kodak/`）：
```
cd src
python eval_literature_metrics.py
```

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

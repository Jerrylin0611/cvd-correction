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

## 使用方式

安裝套件：
```
pip install -r requirements.txt
```

訓練：
```
cd src
python train.py --data_dir "../data/val2017" --epochs 30 --batch_size 8
```

即時 webcam demo（訓練完成後）：
```
cd src
python webcam_demo.py --checkpoint checkpoints/model_epoch30.pt
```
鍵盤操作：`1`/`2`/`3` 切換色弱類型（protanopia/deuteranopia/tritanopia），`q` 離開。
畫面會並排顯示：原始畫面 / 校正前的色弱模擬 / 校正後的色弱模擬，方便直接比較效果。

## 訓練環境

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB)
- 訓練資料: COCO val2017，5000 張圖片，256x256 隨機裁切

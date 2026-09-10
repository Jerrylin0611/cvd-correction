"""
輕量化 U-Net 模型：即時色弱校正網路。

設計理念 (對應跟老師討論的方向):
- 這個任務是「像素級顏色重映射」而不是生成全新畫面內容，
  所以用 U-Net 的 encoder-decoder + skip connection 架構，
  比 GAN 更輕、更穩定，適合即時 webcam 這種要求速度的場景。
- 每一層卷積都換成 depthwise separable convolution
  (MobileNet 系列的核心技巧)：把一般卷積拆成
  「先各 channel 分開做空間卷積」+「再用 1x1 卷積混合 channel」，
  運算量可以降到原本的 1/8~1/9，是達成即時性的關鍵。
- 網路輸出的是「校正量 (residual)」而不是直接輸出整張圖，
  也就是 校正後圖片 = 原圖 + 網路預測的顏色調整量。
  這樣訓練初期網路只要輸出接近 0，就已經是一個「不校正」的合理起點，
  訓練會更穩定、也比較不會讓顏色跑掉太多 (呼應論文提到 GAN 常見的顏色失真問題)。
- 可選的「色弱類型條件輸入」：把要校正的色弱類型 (protan/deuter/tritan)
  編碼成額外的常數 channel 一起餵進網路，這樣同一個模型就能服務
  多種色弱類型，不用每種類型各訓練一個模型。
"""

import torch
import torch.nn as nn

from cvd_simulation import CVD_TYPES

NUM_CVD_TYPES = len(CVD_TYPES)


class DepthwiseSeparableConv(nn.Module):
    """MobileNet 風格的輕量卷積: depthwise (逐 channel 空間卷積) + pointwise (1x1 混合 channel)。"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class DownBlock(nn.Module):
    """Encoder 的一層：一次 stride=2 的輕量卷積負責降解析度，再接一層卷積加深特徵。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = DepthwiseSeparableConv(in_ch, out_ch, stride=2)
        self.conv = DepthwiseSeparableConv(out_ch, out_ch, stride=1)

    def forward(self, x):
        x = self.down(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """Decoder 的一層：升解析度後跟 encoder 對應層的 feature map (skip connection) 接起來，
    skip connection 是 U-Net 保留空間細節/邊緣的關鍵，對顏色校正這種要保留物體輪廓的任務很重要。
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = DepthwiseSeparableConv(in_ch + skip_ch, out_ch, stride=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LightUNetColorCorrector(nn.Module):
    """
    輕量 U-Net，輸入一張 RGB 圖片 (+ 選用的色弱類型條件)，
    輸出跟輸入同尺寸的「校正後 RGB 圖片」。
    """

    def __init__(self, use_cvd_condition: bool = True, base_ch: int = 16):
        super().__init__()
        self.use_cvd_condition = use_cvd_condition
        in_ch = 3 + (NUM_CVD_TYPES if use_cvd_condition else 0)

        # Encoder：解析度每層減半，channel 數每層加倍 (base_ch -> 2x -> 4x -> 8x)
        self.stem = DepthwiseSeparableConv(in_ch, base_ch, stride=1)
        self.down1 = DownBlock(base_ch, base_ch * 2)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)
        self.down3 = DownBlock(base_ch * 4, base_ch * 8)

        # Bottleneck：最深層，解析度最小、特徵最抽象
        self.bottleneck = DepthwiseSeparableConv(base_ch * 8, base_ch * 8, stride=1)

        # Decoder：對稱地把解析度還原回去，並融合 encoder 的 skip feature
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch)

        # 輸出層：1x1 卷積把特徵壓回 3 channel (RGB) 的「校正量」
        # 用 tanh 把校正量限制在 [-1, 1]，避免單次調整過大導致顏色劇烈跳動
        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=1)
        self.out_act = nn.Tanh()

    def _build_condition_map(self, x: torch.Tensor, cvd_type_idx: torch.Tensor) -> torch.Tensor:
        """把每張圖對應的色弱類型 index，展開成跟圖片一樣大小的常數 channel。"""
        b, _, h, w = x.shape
        cond = torch.zeros(b, NUM_CVD_TYPES, h, w, device=x.device, dtype=x.dtype)
        for i, idx in enumerate(cvd_type_idx):
            cond[i, idx] = 1.0
        return cond

    def forward(self, x: torch.Tensor, cvd_type_idx: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, 3, H, W)，範圍 [0,1] 的原始 RGB 圖片
        cvd_type_idx: (B,) 的整數 tensor，指定每張圖要針對哪種色弱類型做校正
                      (index 對應 cvd_simulation.CVD_TYPES 的順序)
        回傳: (B, 3, H, W)，校正後的 RGB 圖片，範圍 [0,1]
        """
        net_input = x
        if self.use_cvd_condition:
            if cvd_type_idx is None:
                raise ValueError("use_cvd_condition=True 時必須提供 cvd_type_idx")
            cond = self._build_condition_map(x, cvd_type_idx)
            net_input = torch.cat([x, cond], dim=1)

        s0 = self.stem(net_input)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)

        b = self.bottleneck(s3)

        d3 = self.up3(b, s2)
        d2 = self.up2(d3, s1)
        d1 = self.up1(d2, s0)

        residual = self.out_act(self.out_conv(d1))

        # 核心設計：輸出 = 原圖 + 校正量，並 clamp 回合法的圖片範圍 [0,1]
        corrected = torch.clamp(x + residual, 0.0, 1.0)
        return corrected

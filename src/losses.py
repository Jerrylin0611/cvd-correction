"""
訓練用的損失函數。

整體 loss 分成兩個目標，剛好對應「即時性方向可行嗎」討論裡提到的兩個評估重點：

1. naturalness (自然度)：校正後的圖片給一般人看，不能失真、不能變得很奇怪。
   這是文獻裡 GAN-based 方法常被詬病的問題 (unnatural/discordant colors)，
   我們用 L1 + SSIM 兩種方式一起約束，比只用 L1 更能保留邊緣結構。

2. distinguishability (可辨識度)：色弱患者看校正後的圖片，
   要比看原圖能分辨出更多本來會搞混的顏色。
   做法是把「原圖的顏色對比」跟「校正後圖片經過 CVD 模擬看到的對比」拿來比較，
   逼迫網路想辦法把色弱視覺下會消失的對比 (用其他顏色/亮度線索) 保留下來。
   這其實就是傳統 daltonization 演算法 (模擬 -> 比對 -> 校正) 的精神，
   差別是這裡讓神經網路自己學怎麼校正，而不是套用一個固定公式。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim

from cvd_simulation import simulate_cvd, simulate_cvd_per_sample

# Sobel 濾波器：用來抓圖片的邊緣/局部對比資訊
_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)


def _gradient_magnitude(img: torch.Tensor) -> torch.Tensor:
    """
    計算圖片的局部對比 (local contrast)。

    這裡刻意「不」轉灰階，而是分別對 R/G/B 三個 channel 各自做 Sobel 再合併。
    原因：色弱患者最大的困擾，是「亮度相近、但色調(hue)不同」的顏色會混淆
    (例如紅綠色盲的經典困擾)。如果先轉灰階，等於直接丟掉色調資訊，
    只留下亮度對比 —— 但亮度對比本來就不太受色弱影響，
    這樣算出來的 loss 幾乎量測不到「色調造成的對比」，
    會導致模型學到「什麼都不校正」也能讓 loss 很低 (訓練實驗證實了這個問題)。
    所以改成在 RGB 三個 channel 上分別算梯度、取平方和開根號合併，
    這樣色調差異造成的 channel 間變化也會被算進「對比」裡。

    img: (B, 3, H, W)，回傳 (B, 1, H, W)
    """
    sobel_x = _SOBEL_X.to(img.device, img.dtype).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    sobel_y = _SOBEL_Y.to(img.device, img.dtype).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    # groups=3: 每個 channel 各自套用同一組 Sobel 核，不互相混合
    gx = F.conv2d(img, sobel_x, padding=1, groups=3)
    gy = F.conv2d(img, sobel_y, padding=1, groups=3)

    # 三個 channel 的梯度平方和開根號，再對 channel 取平均 -> (B, 1, H, W)
    magnitude_per_channel = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    return magnitude_per_channel.mean(dim=1, keepdim=True)


def naturalness_loss(corrected: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """校正後圖片不該跟原圖差太多 (給一般人看的自然度)。L1 對顏色平移比較敏感、好收斂。"""
    return F.l1_loss(corrected, original)


def structure_loss(corrected: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """用 SSIM 補強 L1 沒顧到的結構/邊緣資訊，1 - SSIM 當作 loss (SSIM 越接近1越好)。"""
    return 1.0 - ssim(corrected, original, data_range=1.0, size_average=True)


def distinguishability_loss(
    corrected: torch.Tensor, original: torch.Tensor,
    cvd_types: list = None, cvd_type_idx: torch.Tensor = None,
    scales: tuple = (1, 4, 16),
) -> torch.Tensor:
    """
    核心的「有沒有真的幫到色弱患者」loss。

    對每一種色弱類型：
      1. 把校正後圖片模擬成色弱患者看到的樣子 corrected_sim
      2. 比較 corrected_sim 的邊緣對比 跟 原圖 (全彩、正常視覺) 的邊緣對比
      3. 兩者差越小，代表色弱患者透過校正後圖片，
         感受到的對比越接近正常人看原圖的感受 -> 校正越成功

    傳 cvd_type_idx (每張圖各自對應的色弱類型 index) 時，只用該圖被模型
    條件輸入指定的那一種類型去算，不會混進其他類型的模擬結果——訓練時
    用這個，才能讓模型的校正真的跟它收到的條件輸入對齊 (見 train.py)。
    沒傳 cvd_type_idx 時，改用 cvd_types (list[str])，對整個 batch
    輪流套用每一種類型再平均，只用於評估腳本 (eval_checkpoint.py /
    _compare_versions.py) 想看「整體」表現的情境。

    多尺度 (scales)：Sobel 邊緣偵測只看得到局部交界線，對「一大片純色物體」
    這種內部完全沒有邊緣的區域是死角 (實測發現：純紅 vs 純綠色塊，
    校正後兩色在 CVD 模擬下反而更相近，因為模型可以任意調整色塊內部顏色
    而不影響邊緣 loss)。所以額外把圖片縮小 (avg_pool) 到粗粒度再算一次，
    縮小後原本無邊緣的大色塊會因為跟鄰近區塊平均值不同而產生「邊緣」，
    逼模型也要顧到大面積區域的整體色彩對比，不是只顧細節邊緣。
    """
    total = 0.0
    for scale in scales:
        if scale > 1:
            orig_scaled = F.avg_pool2d(original, kernel_size=scale)
            corrected_scaled = F.avg_pool2d(corrected, kernel_size=scale)
        else:
            orig_scaled = original
            corrected_scaled = corrected

        original_edges = _gradient_magnitude(orig_scaled)

        if cvd_type_idx is not None:
            corrected_sim = simulate_cvd_per_sample(corrected_scaled, cvd_type_idx)
            corrected_sim_edges = _gradient_magnitude(corrected_sim)
            total = total + F.l1_loss(corrected_sim_edges, original_edges)
        else:
            for cvd_type in cvd_types:
                corrected_sim = simulate_cvd(corrected_scaled, cvd_type)
                corrected_sim_edges = _gradient_magnitude(corrected_sim)
                total = total + F.l1_loss(corrected_sim_edges, original_edges)

    n_types = 1 if cvd_type_idx is not None else len(cvd_types)
    return total / (len(scales) * n_types)


def palette_distance_loss(
    corrected: torch.Tensor, original: torch.Tensor,
    cvd_types: list = None, cvd_type_idx: torch.Tensor = None, grid: int = 8,
) -> torch.Tensor:
    """
    補強 distinguishability_loss 的死角：邊緣型 loss 只看得到「相鄰交界處」，
    對「兩塊不相鄰、但都是大面積純色」的區域完全沒轍
    (實測驗證：純紅色塊 vs 純綠色塊，邊緣 loss 沒辦法讓兩者在 CVD 模擬下變得更好分辨)。

    做法：把圖片縮成一個 grid x grid 的「調色盤」(每格是該區域的平均色)，
    不管這些格子在畫面上是否相鄰，直接算「任兩格之間的顏色距離」，
    然後要求：校正後圖片在 CVD 模擬下，這些任兩格的顏色距離
    要盡量跟原圖 (正常視覺) 的任兩格顏色距離一致。
    這樣不管兩個顏色區塊隔多遠、面積多大，只要原本看起來不同，
    校正後在色弱視覺下也要維持不同。

    cvd_type_idx / cvd_types 的用法跟 distinguishability_loss 一致
    (per-sample 條件對齊 vs. 評估腳本用的整體平均)。
    """
    orig_palette = F.adaptive_avg_pool2d(original, (grid, grid))
    corrected_palette = F.adaptive_avg_pool2d(corrected, (grid, grid))

    orig_flat = orig_palette.flatten(2).transpose(1, 2)  # (B, grid*grid, 3)
    orig_dist = torch.cdist(orig_flat, orig_flat)  # (B, N, N) 任兩格的顏色距離 (原圖、正常視覺)

    if cvd_type_idx is not None:
        corrected_sim_palette = simulate_cvd_per_sample(corrected_palette, cvd_type_idx)
        sim_flat = corrected_sim_palette.flatten(2).transpose(1, 2)
        sim_dist = torch.cdist(sim_flat, sim_flat)
        return F.l1_loss(sim_dist, orig_dist)

    total = 0.0
    for cvd_type in cvd_types:
        corrected_sim_palette = simulate_cvd(corrected_palette, cvd_type)
        sim_flat = corrected_sim_palette.flatten(2).transpose(1, 2)
        sim_dist = torch.cdist(sim_flat, sim_flat)
        total = total + F.l1_loss(sim_dist, orig_dist)

    return total / len(cvd_types)


def bias_penalty_loss(corrected: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """
    懲罰「整張圖平均色偏」，避免模型學到最省事的解法：
    對全畫面套一個均勻色調濾鏡 (實驗中觀察到的失敗模式：整張圖偏紫/洋紅)。

    做法：算每張圖、每個 channel 在空間上的平均校正量 (mean over H,W)，
    這個平均值代表的就是「整張圖的色偏方向與強度」。
    只懲罰這個平均值，不會限制模型做局部、針對性的顏色調整
    (因為局部調整有正有負，平均起來還是接近 0)。
    """
    residual = corrected - original
    mean_shift_per_channel = residual.mean(dim=[2, 3])  # (B, 3)
    return (mean_shift_per_channel ** 2).mean()


class ColorCorrectionLoss(nn.Module):
    """把五個 loss 加權合併成一個總 loss，權重之後可以依實驗結果調整。"""

    def __init__(
        self,
        w_natural: float = 0.1,
        w_structure: float = 0.1,
        w_distinguish: float = 10.0,
        w_palette: float = 10.0,
        w_bias: float = 5.0,
    ):
        super().__init__()
        self.w_natural = w_natural
        self.w_structure = w_structure
        self.w_distinguish = w_distinguish
        self.w_palette = w_palette
        self.w_bias = w_bias

    def forward(
        self, corrected: torch.Tensor, original: torch.Tensor,
        cvd_types: list = None, cvd_type_idx: torch.Tensor = None,
    ):
        l_natural = naturalness_loss(corrected, original)
        l_structure = structure_loss(corrected, original)
        l_distinguish = distinguishability_loss(corrected, original, cvd_types, cvd_type_idx)
        l_palette = palette_distance_loss(corrected, original, cvd_types, cvd_type_idx)
        l_bias = bias_penalty_loss(corrected, original)

        total = (
            self.w_natural * l_natural
            + self.w_structure * l_structure
            + self.w_distinguish * l_distinguish
            + self.w_palette * l_palette
            + self.w_bias * l_bias
        )

        # 個別數值回傳出去方便訓練時印出來觀察 (例如發現自然度掉太快、可辨識度沒在學)
        return total, {
            "natural": l_natural.item(),
            "structure": l_structure.item(),
            "distinguish": l_distinguish.item(),
            "palette": l_palette.item(),
            "bias": l_bias.item(),
            "total": total.item(),
        }

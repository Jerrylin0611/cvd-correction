"""
色覺缺陷 (CVD) 模擬模組。

用途：給定一張正常視覺的圖片，模擬色弱/色盲患者實際看到的樣子。
這個模擬結果會在 losses.py 裡被拿來當作「訓練訊號」：
我們讓模型去比較「原圖」跟「校正後圖片經過 CVD 模擬」的差異，
藉此判斷校正是否真的有把色弱患者分辨不出來的顏色資訊找回來。

使用的是 Machado, Oliveira & Fair (2009) 提出的模擬矩陣，
這是目前色覺模擬最常被引用、也是 Hue4U 等最新論文採用的生理模型之一。
矩陣是在「線性 RGB」空間下運作的，所以流程一定是：
sRGB (一般圖片) -> 線性 RGB -> 套用矩陣模擬色盲 -> 轉回 sRGB。

注意 tritanopia：Machado 2009 原作者明說沒有真正去建模 tritanopia (只是 tritanomaly 的
shift 近似)，文獻 (DaltonLens 2021 對開源模擬法的評比) 建議 tritan 改用
Brettel, Viénot & Mollon (1997)。所以另外提供 Brettel 1997 的 tritan 模擬，
用 set_tritan_model("brettel1997") 切換。預設仍是 Machado，訓練行為不變；
目前只拿 Brettel 來「換一個比較可信的模擬重新評估」，見 _verify_brettel_tritan.py。
"""

import torch

# Machado et al. (2009) 二色視覺 (dichromacy, severity=1.0，最嚴重程度) 轉換矩陣
# 每個矩陣都是 3x3，作用在線性 RGB 向量上：C_sim = M @ C_linear
_CVD_MATRICES = {
    "protanopia": torch.tensor([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ]),
    "deuteranopia": torch.tensor([
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ]),
    "tritanopia": torch.tensor([
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900],
    ]),
}

# Brettel, Viénot & Mollon (1997) 的 tritanopia 模擬：在 LMS 空間把顏色投影到兩個
# 半平面 (以 485nm / 660nm 為錨點) 之一，所以是「兩個矩陣 + 依顏色落在哪一側挑一個」。
# 下面數值是用 daltonlens 0.1.5 的 Simulator_Brettel1997 (sRGB, Smith & Pokorny 1975
# 錐細胞基礎、以 RGB 白當中性軸) 換算成「線性 RGB -> 線性 RGB」的形式，
# 對隨機圖片跟 daltonlens 的結果誤差 < 1e-14，白色模擬後仍是白色。
_BRETTEL_TRITAN_M1 = torch.tensor([
    [1.013542, 0.142682, -0.156224],
    [-0.011805, 0.875612, 0.136194],
    [0.077073, 0.812081, 0.110847],
])
_BRETTEL_TRITAN_M2 = torch.tensor([
    [0.933370, 0.199990, -0.133360],
    [0.058087, 0.825652, 0.116261],
    [-0.379228, 1.138250, 0.240978],
])
# 分隔平面的法向量 (已換算到線性 RGB)：dot(rgb, n) < 0 用 M2，否則用 M1
_BRETTEL_TRITAN_SEP = torch.tensor([0.039601, -0.028307, -0.011294])

TRITAN_MODELS = ["machado2009", "brettel1997"]
_tritan_model = "machado2009"


def set_tritan_model(name: str) -> None:
    """切換 tritanopia 要用哪個模擬模型 (影響 simulate_cvd / simulate_cvd_per_sample，
    也就連帶影響 losses.py 的 distinguish/palette)。只有 tritanopia 受影響。"""
    global _tritan_model
    if name not in TRITAN_MODELS:
        raise ValueError(f"tritan 模擬模型只支援 {TRITAN_MODELS}，收到 {name}")
    _tritan_model = name


def _brettel_tritan_linear(linear: torch.Tensor) -> torch.Tensor:
    """線性 RGB (B,3,H,W) -> Brettel 1997 tritanopia 模擬 (線性 RGB，尚未 clamp)。"""
    m1 = _BRETTEL_TRITAN_M1.to(linear.device, linear.dtype)
    m2 = _BRETTEL_TRITAN_M2.to(linear.device, linear.dtype)
    sep = _BRETTEL_TRITAN_SEP.to(linear.device, linear.dtype)
    use_m2 = torch.einsum("c,bchw->bhw", sep, linear).unsqueeze(1) < 0
    return torch.where(
        use_m2,
        torch.einsum("oc,bchw->bohw", m2, linear),
        torch.einsum("oc,bchw->bohw", m1, linear),
    )


# 給模型當「條件輸入」用的固定順序，方便用 index 表示色弱類型
CVD_TYPES = list(_CVD_MATRICES.keys())

# 疊成 (3, 3, 3) 的 tensor，index 對應 CVD_TYPES 的順序，方便用 cvd_type_idx 直接索引，
# 這樣同一個 batch 裡每張圖可以各自套用自己對應的色弱類型矩陣 (見 simulate_cvd_per_sample)
_ALL_MATRICES = torch.stack([_CVD_MATRICES[t] for t in CVD_TYPES])


def srgb_to_linear(img: torch.Tensor) -> torch.Tensor:
    """sRGB (gamma 校正過) -> 線性 RGB。img 範圍需為 [0,1]，shape 為 (..., 3, H, W)。"""
    # sRGB 的標準反 gamma 公式，分段是因為在接近黑色時單純次方會不準確
    threshold = 0.04045
    low = img / 12.92
    high = ((img + 0.055) / 1.055) ** 2.4
    return torch.where(img <= threshold, low, high)


def linear_to_srgb(img: torch.Tensor) -> torch.Tensor:
    """線性 RGB -> sRGB。是 srgb_to_linear 的反函數。"""
    threshold = 0.0031308
    low = img * 12.92
    # 這裡的 pow 指數 1/2.4 < 1，在輸入趨近 0 時梯度會趨近無限大。
    # torch.where 雖然在 forward pass 只挑一個分支的「值」，
    # 但 backward pass 兩個分支的梯度都會被算出來 (只是沒被選到的那支會乘上0)，
    # 如果沒選到的分支剛好梯度是 inf，就會出現 0 * inf = NaN，把權重整組污染掉。
    # 所以這裡刻意 clamp 到一個很小的正數而不是 0，讓 pow 的梯度保持有限。
    high = 1.055 * torch.clamp(img, min=1e-8).pow(1 / 2.4) - 0.055
    return torch.where(img <= threshold, low, high)


def simulate_cvd(img: torch.Tensor, cvd_type: str) -> torch.Tensor:
    """
    模擬色弱患者看到的畫面。

    img: shape (B, 3, H, W)，數值範圍 [0,1] 的 sRGB 圖片 (一般相機/螢幕格式)
    cvd_type: "protanopia" / "deuteranopia" / "tritanopia" 其中一種
    回傳: 同樣 shape，模擬後的 sRGB 圖片
    """
    linear = srgb_to_linear(img)
    if cvd_type == "tritanopia" and _tritan_model == "brettel1997":
        simulated_linear = _brettel_tritan_linear(linear)
    else:
        matrix = _CVD_MATRICES[cvd_type].to(img.device, img.dtype)
        # 用 einsum 把矩陣套用在 channel 維度上 (B,3,H,W) -> (B,3,H,W)
        simulated_linear = torch.einsum("oc,bchw->bohw", matrix, linear)
    simulated_linear = torch.clamp(simulated_linear, 0.0, 1.0)

    return linear_to_srgb(simulated_linear)


def simulate_cvd_per_sample(img: torch.Tensor, cvd_type_idx: torch.Tensor) -> torch.Tensor:
    """
    跟 simulate_cvd 一樣，差別是每張圖可以各自指定不同的色弱類型
    (而不是整個 batch 套同一種)，用來讓訓練 loss 對齊模型當下實際被
    條件輸入指定的那個色弱類型，而不是每張圖都跟三種類型的模擬結果比對。

    img: shape (B, 3, H, W)，數值範圍 [0,1] 的 sRGB 圖片
    cvd_type_idx: shape (B,) 的 long tensor，每個元素對應 CVD_TYPES 的 index
    回傳: 同樣 shape，模擬後的 sRGB 圖片
    """
    matrices = _ALL_MATRICES.to(img.device, img.dtype)[cvd_type_idx]  # (B, 3, 3)

    linear = srgb_to_linear(img)
    simulated_linear = torch.einsum("boc,bchw->bohw", matrices, linear)
    if _tritan_model == "brettel1997":
        is_tritan = (cvd_type_idx == CVD_TYPES.index("tritanopia")).view(-1, 1, 1, 1)
        if is_tritan.any():
            simulated_linear = torch.where(is_tritan, _brettel_tritan_linear(linear), simulated_linear)
    simulated_linear = torch.clamp(simulated_linear, 0.0, 1.0)

    return linear_to_srgb(simulated_linear)

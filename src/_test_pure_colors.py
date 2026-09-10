"""
用乾淨的合成色塊（純紅 vs 純綠）測試模型的校正力道，排除攝影機光線/白平衡的干擾。
直接量化「校正前」跟「校正後」在色弱模擬下，兩個顏色到底差多少。
"""

import numpy as np
import torch
from PIL import Image

from cvd_simulation import CVD_TYPES, simulate_cvd
from model import LightUNetColorCorrector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 產生一張圖：左半純紅 (255,0,0)，右半純綠 (0,255,0)
size = 256
img = np.zeros((size, size, 3), dtype=np.uint8)
img[:, : size // 2] = [255, 0, 0]
img[:, size // 2 :] = [0, 255, 0]
img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
img_tensor = img_tensor.to(device)

model = LightUNetColorCorrector(use_cvd_condition=True).to(device)
model.load_state_dict(torch.load("checkpoints/model_epoch50.pt", map_location=device))
model.eval()

cvd_type = "deuteranopia"
cvd_idx = torch.tensor([CVD_TYPES.index(cvd_type)], device=device)

with torch.no_grad():
    corrected = model(img_tensor, cvd_idx)
    before_sim = simulate_cvd(img_tensor, cvd_type)
    after_sim = simulate_cvd(corrected, cvd_type)


def sample_colors(t):
    """取左半跟右半各自的平均顏色 (RGB, 0~255)。"""
    left = t[0, :, :, : size // 2].mean(dim=[1, 2]).cpu().numpy() * 255
    right = t[0, :, :, size // 2 :].mean(dim=[1, 2]).cpu().numpy() * 255
    return left, right


def luminance(rgb):
    """感知亮度 (跟人眼對亮度的感知比較接近的加權方式)。"""
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


for name, tensor in [("原圖(未模擬)", img_tensor), ("校正前+CVD模擬", before_sim), ("校正後+CVD模擬", after_sim)]:
    left, right = sample_colors(tensor)
    diff = np.linalg.norm(left - right)
    lum_diff = abs(luminance(left) - luminance(right))
    print(f"[{name}] 左(原紅)={left.round(1)} 右(原綠)={right.round(1)}  "
          f"歐式色差={diff:.1f}  亮度差={lum_diff:.1f}")

# 存圖方便直接用眼睛看
def to_img(t):
    arr = (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return arr

grid = np.vstack([to_img(img_tensor), to_img(before_sim), to_img(after_sim)])
Image.fromarray(grid).save("../pure_color_test.png")
print("已存圖: pure_color_test.png (由上到下: 原圖 / 校正前色弱模擬 / 校正後色弱模擬)")

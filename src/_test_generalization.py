"""
驗證 V5 的純色測試暴漲(118->189)是不是洗分：ConfusionPairBlocks 訓練時用的顏色對
跟版面，跟 eval_checkpoint.py 的測試圖案幾乎一樣 (1x2 對半、飽和紅/綠)。

這裡故意換一組「訓練時沒出現過」的顏色對 + 版面 (3x3 棋盤格，色調也不是滿飽和度)，
分別測 V2 跟 V5，看 V5 的優勢是不是只在原測試圖案上才有。

用法:
    python _test_generalization.py
"""

import numpy as np
import torch
from PIL import Image

from cvd_simulation import CVD_TYPES, simulate_cvd
from model import LightUNetColorCorrector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = [
    ("V2", "checkpoints_v2/model_epoch50.pt"),
    ("V5", "checkpoints_v5/model_epoch20.pt"),
]

# 訓練時 ConfusionPairBlocks 只用過飽和度 0/1 的顏色 + 1x2/2x1/2x2 版面。
# 這裡故意都不重複：3x3 棋盤格、去飽和的中間色調。
HELD_OUT_CASES = [
    ("deuteranopia", (0.80, 0.40, 0.40), (0.40, 0.80, 0.40), "去飽和紅/綠"),
    ("protanopia", (0.75, 0.45, 0.20), (0.30, 0.55, 0.20), "棕橘/橄欖綠"),
    ("tritanopia", (0.20, 0.20, 0.90), (0.90, 0.90, 0.30), "天藍/芥黃"),
]


def load_model(path):
    model = LightUNetColorCorrector(use_cvd_condition=True).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def make_checkerboard(size, color_a, color_b, grid=3):
    img = torch.zeros(3, size, size)
    bounds = torch.linspace(0, size, grid + 1).round().long()
    mask_a = torch.zeros(size, size, dtype=torch.bool)
    for r in range(grid):
        for c in range(grid):
            color = color_a if (r + c) % 2 == 0 else color_b
            img[:, bounds[r]:bounds[r + 1], bounds[c]:bounds[c + 1]] = torch.tensor(color).view(3, 1, 1)
            if (r + c) % 2 == 0:
                mask_a[bounds[r]:bounds[r + 1], bounds[c]:bounds[c + 1]] = True
    return img, mask_a


def region_diff(sim_img, mask_a):
    flat = sim_img[0]  # (3, H, W)
    mean_a = flat[:, mask_a].mean(dim=1).cpu().numpy() * 255
    mean_b = flat[:, ~mask_a].mean(dim=1).cpu().numpy() * 255
    return float(np.linalg.norm(mean_a - mean_b))


def to_img(t):
    return (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


print("=" * 70)
print("held-out 顏色對 + 3x3 棋盤格版面 (訓練時都沒出現過)")
print("=" * 70)

grid_rows = []
row_labels = []

for cvd_type, color_a, color_b, desc in HELD_OUT_CASES:
    size = 256
    img, mask_a = make_checkerboard(size, color_a, color_b)
    img_tensor = img.unsqueeze(0).to(device)
    cvd_idx = torch.tensor([CVD_TYPES.index(cvd_type)], device=device)

    before_sim = simulate_cvd(img_tensor, cvd_type)
    diff_before = region_diff(before_sim, mask_a)
    print(f"\n[{cvd_type}] {desc}  色差(未校正)={diff_before:.1f}")

    row = [to_img(img_tensor), to_img(before_sim)]
    labels = ["原圖", "未校正+CVD模擬"]

    for tag, path in CHECKPOINTS:
        model = load_model(path)
        with torch.no_grad():
            corrected = model(img_tensor, cvd_idx)
            after_sim = simulate_cvd(corrected, cvd_type)
        diff_after = region_diff(after_sim, mask_a)
        verdict = "改善" if diff_after > diff_before else "變差"
        print(f"  [{tag}] 校正後色差={diff_after:.1f}  ({diff_before:.1f} -> {diff_after:.1f}, {verdict})")
        row.append(to_img(after_sim))
        labels.append(f"{tag}校正後")

    grid_rows.append(np.hstack(row))
    row_labels.append(labels)

grid = np.vstack(grid_rows)
Image.fromarray(grid).save("../generalization_test.png")
print(f"\n已存圖: generalization_test.png")
print("每列由左到右: " + " / ".join(row_labels[0]))

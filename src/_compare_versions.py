"""
比較多個 checkpoint 版本的校正效果，公平地放在同一張圖上看:
1. 純紅/純綠合成色塊測試 (數值 + 存圖)
2. 真實照片範例 (固定 seed，所有版本用同一批照片，才能公平比較)

用法:
    python _compare_versions.py
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from cvd_simulation import CVD_TYPES, simulate_cvd
from dataset import ColorImageFolder
from model import LightUNetColorCorrector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = [
    ("OLD", "checkpoints/model_epoch50.pt"),
    ("V2", "checkpoints_v2/model_epoch50.pt"),
    ("V3", "checkpoints_v3/model_epoch50.pt"),
    ("V4", "checkpoints_v4/model_epoch50.pt"),
]

cvd_type = "deuteranopia"
cvd_idx_val = CVD_TYPES.index(cvd_type)


def load_model(path):
    model = LightUNetColorCorrector(use_cvd_condition=True).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def sample_halves(t, size):
    left = t[0, :, :, : size // 2].mean(dim=[1, 2]).cpu().numpy() * 255
    right = t[0, :, :, size // 2 :].mean(dim=[1, 2]).cpu().numpy() * 255
    return left, right


def to_img(t):
    return (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def to_numpy_batch(t):
    return (t.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


print("=" * 60)
print("1. 純紅/純綠色塊測試 (deuteranopia)")
print("=" * 60)

size = 256
raw = np.zeros((size, size, 3), dtype=np.uint8)
raw[:, : size // 2] = [255, 0, 0]
raw[:, size // 2 :] = [0, 255, 0]
img_tensor = torch.from_numpy(raw).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
cvd_idx = torch.tensor([cvd_idx_val], device=device)

before_sim = simulate_cvd(img_tensor, cvd_type)
l, r = sample_halves(before_sim, size)
diff_before = float(np.linalg.norm(l - r))
print(f"[校正前]        色差={diff_before:.1f}")

rows = [to_img(img_tensor), to_img(before_sim)]
labels = ["原圖(未模擬)", "校正前+CVD模擬"]

for tag, path in CHECKPOINTS:
    model = load_model(path)
    with torch.no_grad():
        corrected = model(img_tensor, cvd_idx)
        after_sim = simulate_cvd(corrected, cvd_type)
    l, r = sample_halves(after_sim, size)
    diff_after = float(np.linalg.norm(l - r))
    print(f"[{tag:<4}校正後]    色差={diff_after:.1f}   (改善量: {diff_before:.1f} -> {diff_after:.1f})")
    rows.append(to_img(after_sim))
    labels.append(f"{tag} 校正後")

grid = np.vstack(rows)
Image.fromarray(grid).save("../pure_color_compare.png")
print(f"\n已存圖: pure_color_compare.png")
print("由上到下: " + " / ".join(labels))

print()
print("=" * 60)
print("2. 真實照片範例 (固定 seed，所有版本同一批照片)")
print("=" * 60)

torch.manual_seed(0)
dataset = ColorImageFolder("../data/val2017", image_size=256)
loader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=0)
images, _ = next(iter(loader))
images = images.to(device)
cvd_type_idx = torch.full((images.shape[0],), cvd_idx_val, device=device, dtype=torch.long)

orig_np = to_numpy_batch(images)
before_sim_np = to_numpy_batch(simulate_cvd(images, cvd_type))

col_blocks = [orig_np, before_sim_np]
col_labels = ["原圖", "校正前色弱模擬"]

for tag, path in CHECKPOINTS:
    model = load_model(path)
    with torch.no_grad():
        corrected = model(images, cvd_type_idx)
        after_sim = simulate_cvd(corrected, cvd_type)
    col_blocks.append(to_numpy_batch(after_sim))
    col_labels.append(f"{tag}校正後模擬")

grid_rows = []
for i in range(images.shape[0]):
    row = np.hstack([blk[i] for blk in col_blocks])
    grid_rows.append(row)
grid = np.vstack(grid_rows)
Image.fromarray(grid).save("../real_photo_compare.png")
print(f"已存圖: real_photo_compare.png")
print("每列由左到右: " + " / ".join(col_labels))

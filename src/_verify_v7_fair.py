"""
公平重算 V2 vs V7 的 distinguish/palette 分數：用每張圖『實際被指定』的色弱類型
(cvd_type_idx-aligned) 去算，而不是 eval_checkpoint.py 原本『三種都算再平均』的方式
(這種方式對 V7 這種條件式校正的模型不公平，results.csv 裡 V6/V7 看起來變差就是這個原因)。

同時針對三種色弱類型各自存一張真實照片的視覺比較圖，方便直接用眼睛檢查。
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from cvd_simulation import CVD_TYPES, simulate_cvd_per_sample
from dataset import ColorImageFolder
from losses import distinguishability_loss, palette_distance_loss
from model import LightUNetColorCorrector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = [("V2", "checkpoints_v2/model_epoch50.pt"), ("V7", "checkpoints_v7/model_epoch50.pt")]


def load_model(path):
    model = LightUNetColorCorrector(use_cvd_condition=True).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def to_numpy_batch(t):
    return (t.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


torch.manual_seed(0)
dataset = ColorImageFolder("../data/val2017", image_size=256)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
images, _ = next(iter(loader))
images = images.to(device)

models = {tag: load_model(path) for tag, path in CHECKPOINTS}

print("=" * 80)
print("per-sample cvd_type_idx 對齊後的 distinguish / palette loss (越低越好)")
print("=" * 80)

for cvd_type in CVD_TYPES:
    idx = torch.full((images.shape[0],), CVD_TYPES.index(cvd_type), device=device, dtype=torch.long)

    dist_before = distinguishability_loss(images, images, cvd_type_idx=idx).item()
    pal_before = palette_distance_loss(images, images, cvd_type_idx=idx).item()
    print(f"\n[{cvd_type}] 不校正: distinguish={dist_before:.5f}  palette={pal_before:.5f}")

    for tag, model in models.items():
        with torch.no_grad():
            corrected = model(images, idx)
        dist_after = distinguishability_loss(corrected, images, cvd_type_idx=idx).item()
        pal_after = palette_distance_loss(corrected, images, cvd_type_idx=idx).item()
        d_pct = (dist_before - dist_after) / dist_before * 100
        p_pct = (pal_before - pal_after) / pal_before * 100
        print(f"  [{tag}] distinguish={dist_after:.5f} ({d_pct:+.1f}%)  palette={pal_after:.5f} ({p_pct:+.1f}%)")

# 視覺比較圖：每種色弱類型各存一張 (原圖 / 校正前模擬 / V2校正後模擬 / V7校正後模擬)
print("\n存視覺比較圖...")
n_show = 4
for cvd_type in CVD_TYPES:
    idx = torch.full((n_show,), CVD_TYPES.index(cvd_type), device=device, dtype=torch.long)
    imgs = images[:n_show]

    before_sim = simulate_cvd_per_sample(imgs, idx)
    cols = [to_numpy_batch(imgs), to_numpy_batch(before_sim)]
    for tag, model in models.items():
        with torch.no_grad():
            corrected = model(imgs, idx)
        after_sim = simulate_cvd_per_sample(corrected, idx)
        cols.append(to_numpy_batch(after_sim))

    rows = [np.hstack([col[i] for col in cols]) for i in range(n_show)]
    grid = np.vstack(rows)
    out_path = f"../verify_v7_{cvd_type}.png"
    Image.fromarray(grid).save(out_path)
    print(f"  已存: {out_path} (由左到右: 原圖 / 校正前模擬 / V2校正後模擬 / V7校正後模擬)")

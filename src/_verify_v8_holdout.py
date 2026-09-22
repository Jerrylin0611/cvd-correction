"""
在真正沒訓練過的 held-out test split (500 張) 上比較 V2 / V7 / V8：
- V2、V7 是切分機制加入前訓練的，這 500 張其實也在它們的訓練集裡 (不是乾淨測試)。
- V8 是唯一一個訓練時沒看過這 500 張圖的版本 (只用 train split 4500 張訓練，見
  train_v8.log)，所以 V8 在這批圖上的表現才是有意義的「泛化能力」證據：
  如果 V8 在從沒看過的圖上，表現跟 V2/V7 在它們背過的圖上差不多，
  就有力反駁「V7 只是背答案、數據好看」的疑慮。

用法: python _verify_v8_holdout.py
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

CHECKPOINTS = [
    ("V2", "checkpoints_v2/model_epoch50.pt"),
    ("V7", "checkpoints_v7/model_epoch50.pt"),
    ("V8(held-out)", "checkpoints_v8/model_epoch50.pt"),
]


def load_model(path):
    model = LightUNetColorCorrector(use_cvd_condition=True).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def to_numpy_batch(t):
    return (t.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


torch.manual_seed(0)
dataset = ColorImageFolder("../data/val2017", image_size=256, split="test")
print(f"held-out test split 圖片數: {len(dataset)}（V2/V7 訓練時也看過這些圖；V8 沒看過）")
BATCH = 32
loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, num_workers=0)
all_images = [batch.to(device) for batch, _ in loader]  # 分批放到 GPU，避免 500 張一次算 OOM

models = {tag: load_model(path) for tag, path in CHECKPOINTS}


def weighted_avg(values, weights):
    values, weights = np.array(values), np.array(weights)
    return float((values * weights).sum() / weights.sum())


print("=" * 80)
print("在 500 張 held-out 測試圖上，per-sample cvd_type_idx 對齊的 distinguish/palette loss (越低越好)")
print("=" * 80)

for cvd_type in CVD_TYPES:
    dist_before_list, pal_before_list, weights = [], [], []
    dist_after_list = {tag: [] for tag in models}
    pal_after_list = {tag: [] for tag in models}

    for images in all_images:
        idx = torch.full((images.shape[0],), CVD_TYPES.index(cvd_type), device=device, dtype=torch.long)
        weights.append(images.shape[0])
        dist_before_list.append(distinguishability_loss(images, images, cvd_type_idx=idx).item())
        pal_before_list.append(palette_distance_loss(images, images, cvd_type_idx=idx).item())

        for tag, model in models.items():
            with torch.no_grad():
                corrected = model(images, idx)
            dist_after_list[tag].append(distinguishability_loss(corrected, images, cvd_type_idx=idx).item())
            pal_after_list[tag].append(palette_distance_loss(corrected, images, cvd_type_idx=idx).item())

    dist_before = weighted_avg(dist_before_list, weights)
    pal_before = weighted_avg(pal_before_list, weights)
    print(f"\n[{cvd_type}] 不校正: distinguish={dist_before:.5f}  palette={pal_before:.5f}")

    for tag in models:
        dist_after = weighted_avg(dist_after_list[tag], weights)
        pal_after = weighted_avg(pal_after_list[tag], weights)
        d_pct = (dist_before - dist_after) / dist_before * 100
        p_pct = (pal_before - pal_after) / pal_before * 100
        print(f"  [{tag:<13}] distinguish={dist_after:.5f} ({d_pct:+.1f}%)  palette={pal_after:.5f} ({p_pct:+.1f}%)")

print("\n存視覺比較圖 (從 held-out 500 張裡取前 4 張)...")
n_show = 4
for cvd_type in CVD_TYPES:
    idx = torch.full((n_show,), CVD_TYPES.index(cvd_type), device=device, dtype=torch.long)
    imgs = all_images[0][:n_show]

    before_sim = simulate_cvd_per_sample(imgs, idx)
    cols = [to_numpy_batch(imgs), to_numpy_batch(before_sim)]
    for tag, model in models.items():
        with torch.no_grad():
            corrected = model(imgs, idx)
        after_sim = simulate_cvd_per_sample(corrected, idx)
        cols.append(to_numpy_batch(after_sim))

    rows = [np.hstack([col[i] for col in cols]) for i in range(n_show)]
    grid = np.vstack(rows)
    out_path = f"../verify_v8_holdout_{cvd_type}.png"
    Image.fromarray(grid).save(out_path)
    print(f"  已存: {out_path} (由左到右: 原圖 / 校正前模擬 / V2 / V7 / V8(held-out) 校正後模擬)")

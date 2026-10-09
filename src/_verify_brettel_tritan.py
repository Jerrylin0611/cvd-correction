"""
用 Brettel 1997 的 tritanopia 模擬重新評估 V8 / V9 的可辨識度，檢查
「tritan 校正效果」是不是只是 Machado 2009 tritan 矩陣不準造成的假象。

背景 (老師 10/8 建議第 5 點：需要文獻指出藍黃是否效果最差)：
- 文獻沒有直接證據說 tritan 校正最差，但有共識是 Machado 2009 的 tritan 模擬
  不可靠 (原作者沒有真正建模 tritanopia)，tritan 應該用 Brettel 1997。
- 我們的模型訓練時的 loss 用的是 Machado。如果換成比較可信的 Brettel 來評估，
  校正後的可辨識度改善還在，代表模型學到的 tritan 校正是真的有幫助，
  不是只對 Machado 這個特定 (不準的) 模擬有效。

注意：eval_metrics.py 的保真度指標 (SSIM/LPIPS/PCDM) 是「原圖 vs 校正後」，
根本不經過色弱模擬，換模擬模型不會影響那些數字；受影響的只有這裡的
distinguish/palette (可辨識度)。

另外也算「同一張圖，Machado 跟 Brettel 的 tritan 模擬結果差多少」(CIEDE2000)，
當作兩個模擬模型本身落差有多大的參考，protan/deuter 不受影響所以不算。

用法: python _verify_brettel_tritan.py
"""

import sys

import numpy as np
import torch
from skimage.color import deltaE_ciede2000, rgb2lab
from torch.utils.data import DataLoader

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import cvd_simulation
from cvd_simulation import CVD_TYPES, simulate_cvd
from dataset import ColorImageFolder
from losses import distinguishability_loss, palette_distance_loss
from model import LightUNetColorCorrector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = [
    ("V8(對照)", "checkpoints_v8/model_epoch50.pt"),
    ("V9(主模型)", "checkpoints_v9/model_epoch50.pt"),
]
TRITAN = CVD_TYPES.index("tritanopia")


def load_model(path):
    model = LightUNetColorCorrector(use_cvd_condition=True).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def weighted_avg(values, weights):
    values, weights = np.array(values), np.array(weights)
    return float((values * weights).sum() / weights.sum())


dataset = ColorImageFolder("../data/val2017", image_size=256, split="test")
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
all_images = [batch.to(device) for batch, _ in loader]
models = {tag: load_model(path) for tag, path in CHECKPOINTS}

# 校正結果跟模擬模型無關 (模型推論不經過模擬)，先算好兩種模擬共用
with torch.no_grad():
    corrected = {
        tag: [m(imgs, torch.full((imgs.shape[0],), TRITAN, device=device)) for imgs in all_images]
        for tag, m in models.items()
    }

print("tritanopia 可辨識度 (distinguish/palette loss，越低越好，held-out 500 張)")
for sim_model in cvd_simulation.TRITAN_MODELS:
    cvd_simulation.set_tritan_model(sim_model)
    weights, dist_before, pal_before = [], [], []
    dist_after = {tag: [] for tag in models}
    pal_after = {tag: [] for tag in models}

    with torch.no_grad():
        for b, images in enumerate(all_images):
            idx = torch.full((images.shape[0],), TRITAN, device=device, dtype=torch.long)
            weights.append(images.shape[0])
            dist_before.append(distinguishability_loss(images, images, cvd_type_idx=idx).item())
            pal_before.append(palette_distance_loss(images, images, cvd_type_idx=idx).item())
            for tag in models:
                corr = corrected[tag][b]
                dist_after[tag].append(distinguishability_loss(corr, images, cvd_type_idx=idx).item())
                pal_after[tag].append(palette_distance_loss(corr, images, cvd_type_idx=idx).item())

    d0, p0 = weighted_avg(dist_before, weights), weighted_avg(pal_before, weights)
    print(f"\n[模擬模型 {sim_model}] 不校正: distinguish={d0:.5f}  palette={p0:.5f}")
    for tag in models:
        d1, p1 = weighted_avg(dist_after[tag], weights), weighted_avg(pal_after[tag], weights)
        print(
            f"  [{tag:<10}] distinguish={d1:.5f} ({(d0 - d1) / d0 * 100:+.1f}%)  "
            f"palette={p1:.5f} ({(p0 - p1) / p0 * 100:+.1f}%)"
        )

# 兩個模擬模型本身對同一張原圖的 tritan 模擬差多少
diffs = []
with torch.no_grad():
    for images in all_images:
        cvd_simulation.set_tritan_model("machado2009")
        mach = simulate_cvd(images, "tritanopia").permute(0, 2, 3, 1).cpu().numpy()
        cvd_simulation.set_tritan_model("brettel1997")
        bret = simulate_cvd(images, "tritanopia").permute(0, 2, 3, 1).cpu().numpy()
        diffs += [float(np.mean(deltaE_ciede2000(rgb2lab(a), rgb2lab(c)))) for a, c in zip(mach, bret)]
cvd_simulation.set_tritan_model("machado2009")
print(f"\nMachado vs Brettel tritan 模擬結果的差異 (原圖，CIEDE2000 全圖平均)：{np.mean(diffs):.2f} ± {np.std(diffs):.2f}")

"""
同一張圖、三種色弱，V9 vs V11 的校正效果並排比較圖 (給老師看用)。

每一列一種色弱，由左到右：原圖 / 色弱看原圖 / V9 校正後 / 色弱看 V9 / V11 校正後 / 色弱看 V11。
「色弱看」的那幾格標上這張圖的 GCD 補回 % (定義同 eval_literature_metrics.py：
(校正後 - 不校正) / (原圖正常視覺 - 不校正)，CIELAB，固定 seed 抽 512 像素)。
tritan 用 Brettel 1997 模擬 (比 Machado 可信，見 README tritan 一節)；模型本身是用 Machado 訓練的。

預設用 Kodak kodim23 (鸚鵡，紅綠黃藍都有，沒參與訓練)。
用法: python _compare_v9_v11_image.py [圖片路徑] [輸出檔名]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import cvd_simulation
from cvd_simulation import CVD_TYPES, rgb_to_lab, simulate_cvd
from model import LightUNetColorCorrector

image_path = sys.argv[1] if len(sys.argv) > 1 else "../data/kodak/kodim23.png"
out_path = sys.argv[2] if len(sys.argv) > 2 else "../compare_v9_v11.png"
CHECKPOINTS = [("V9", "checkpoints_v9/model_epoch50.pt"), ("V11", "checkpoints_v11/model_epoch50.pt")]
ROWS = [
    ("protanopia", "machado2009", "紅色弱 (protan)"),
    ("deuteranopia", "machado2009", "綠色弱 (deutan)"),
    ("tritanopia", "brettel1997", "藍黃色弱 (tritan, Brettel)"),
]
INFER_SIZE = (256, 384)  # (H, W)，Kodak 是 2:3 橫圖；跟評估一樣在 256 解析度附近推論
GCD_SAMPLES = 512

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img = Image.open(image_path).convert("RGB")
if img.width < img.height:
    img = img.rotate(90, expand=True)
img = img.resize(INFER_SIZE[::-1], Image.BICUBIC)
x = torch.from_numpy(np.asarray(img)).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device)

models = {}
for tag, path in CHECKPOINTS:
    m = LightUNetColorCorrector(use_cvd_condition=True).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    models[tag] = m.eval()

gen = torch.Generator().manual_seed(0)
sample_idx = torch.randperm(INFER_SIZE[0] * INFER_SIZE[1], generator=gen)[:GCD_SAMPLES].to(device)


def gcd(t):
    p = rgb_to_lab(t).flatten(2)[:, :, sample_idx].transpose(1, 2)
    return torch.cdist(p, p).mean().item()


def to_np(t):
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()


fig, axes = plt.subplots(len(ROWS), 6, figsize=(18, 2.35 * len(ROWS) + 0.6))
col_titles = ["原圖", "色弱看原圖", "V9 校正後", "色弱看 V9", "V11 校正後", "色弱看 V11"]
for r, (cvd_type, tritan_model, label) in enumerate(ROWS):
    cvd_simulation.set_tritan_model(tritan_model)
    idx = torch.tensor([CVD_TYPES.index(cvd_type)], device=device)
    with torch.no_grad():
        before = simulate_cvd(x, cvd_type)
        d_orig, d_before = gcd(x), gcd(before)
        panels = [(x, ""), (before, "")]
        for tag, m in models.items():
            out = m(x, idx).clamp(0, 1)
            sim = simulate_cvd(out, cvd_type)
            rec = (gcd(sim) - d_before) / (d_orig - d_before) * 100
            panels += [(out, ""), (sim, f"GCD 補回 {rec:+.1f}%")]
            print(f"{label:<28}{tag:<4} GCD 補回 {rec:+6.1f}%")
    for c, (t, note) in enumerate(panels):
        ax = axes[r, c]
        ax.imshow(to_np(t))
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if r == 0:
            ax.set_title(col_titles[c], fontsize=12)
        if note:
            ax.text(0.02, 0.04, note, transform=ax.transAxes, fontsize=10, color="white",
                    bbox=dict(facecolor="black", alpha=0.6, pad=2, edgecolor="none"))
    axes[r, 0].set_ylabel(label, fontsize=12)
cvd_simulation.set_tritan_model("machado2009")

fig.suptitle("V9 vs V11：同一張圖 (Kodak kodim23，未參與訓練) 對三種色弱的校正效果", fontsize=13, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(out_path, dpi=130)
print(f"已存：{out_path}")

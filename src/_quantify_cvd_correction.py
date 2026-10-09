"""
用影像量化校正效果 (老師 10/8 建議第 4 點)：
在「色弱者會混淆的顏色對」上，算校正前 / 校正後在色弱模擬下兩色的 CIEDE2000 色差，
ΔE 越大代表色弱者越分得出來。兩種測試圖：

1. 混淆色對色塊：左半顏色 A、右半顏色 B (跟 _test_pure_colors.py 同概念，擴充成三類型多組)。
2. 石原式檢查圖：圓點拼成的圓盤，數字的點用顏色 A、背景的點用顏色 B，每個點亮度隨機
   抖動 (跟真的石原圖一樣，避免只靠亮度就認出數字)。量「數字 vs 背景」的平均色差。

混淆色對不是手挑的，而是自動搜尋：在 13^3 的 RGB 網格上找「正常視覺 ΔE > 25、
色弱模擬後 ΔE < 3」的顏色對，再貪婪挑 8 組彼此不重複的。
注意：V9 訓練時用過 synthetic_colors._CONFUSION_PAIRS 的混淆色塊，所以這裡把
跟那些訓練顏色太接近 (ΔE < 15) 的顏色全部排除，避免考到練過的題目。

tritanopia 分兩組：用 Machado 2009 找的混淆對 (用 Machado 量)、用 Brettel 1997 找的
混淆對 (用 Brettel 量)。Machado 的 tritan 模擬太弱，幾乎找不到強混淆對，
見 README「tritanopia（藍黃色弱）的限制」。

輸出：
  ../confusion_pairs_results.csv  每組色對/每張檢查圖的數字
  ../quantify_pairs.png           色對 ΔE 長條圖
  ../quantify_plates.png          檢查圖範例 (原圖 / 校正前模擬 / 校正後模擬)

用法: python _quantify_cvd_correction.py
"""

import csv
import sys

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.color import deltaE_ciede2000, rgb2lab

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import cvd_simulation
from cvd_simulation import CVD_TYPES, simulate_cvd
from model import LightUNetColorCorrector
from synthetic_colors import _CONFUSION_PAIRS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = [
    ("V9", "checkpoints_v9/model_epoch50.pt"),
    ("V8", "checkpoints_v8/model_epoch50.pt"),
]
# (顯示名稱, 色弱類型, tritan 模擬模型)
PAIR_SETS = [
    ("protan", "protanopia", "machado2009"),
    ("deutan", "deuteranopia", "machado2009"),
    ("tritan (Machado)", "tritanopia", "machado2009"),
    ("tritan (Brettel)", "tritanopia", "brettel1997"),
]
PAIRS_PER_SET = 8
NORMAL_MIN, SIM_MAX, TRAIN_EXCLUDE = 25.0, 3.0, 15.0
DIGITS = "2357689412"

# dataviz 預設色盤：校正前用中性灰，V9(主模型) 藍、V8(對照) 橘
COLORS = {"before": "#a3a29b", "V9": "#2a78d6", "V8": "#eb6834"}


def lab(rgb):
    return rgb2lab(np.asarray(rgb, dtype=np.float64).reshape(-1, 1, 3)).reshape(-1, 3)


def de(a, b):
    return float(deltaE_ciede2000(lab(a)[0], lab(b)[0]))


def to_tensor(img):
    return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)


def to_np(t):
    return t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def find_pairs(cvd_type):
    g = np.linspace(0, 1, 13)
    grid = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    l0 = lab(grid)
    train = np.array([c for pairs in _CONFUSION_PAIRS.values() for p in pairs for c in p])
    far = deltaE_ciede2000(l0[:, None], lab(train)[None]).min(axis=1) > TRAIN_EXCLUDE
    grid, l0 = grid[far], l0[far]

    with torch.no_grad():
        sim = simulate_cvd(torch.tensor(grid.T, dtype=torch.float32).reshape(1, 3, -1, 1), cvd_type)
    ls = lab(sim[0, :, :, 0].T.numpy())
    dn = deltaE_ciede2000(l0[:, None], l0[None])
    ds = deltaE_ciede2000(ls[:, None], ls[None])
    i, j = np.where(np.triu((dn > NORMAL_MIN) & (ds < SIM_MAX)))
    order = np.argsort(-dn[i, j])  # 正常視覺差越多的越先挑 (越「該分得出來」)

    chosen, used = [], []
    for k in order:
        a, b = l0[i[k]], l0[j[k]]
        if used and min(deltaE_ciede2000(np.array(used), np.broadcast_to(c, (len(used), 3))).min() for c in (a, b)) < 12:
            continue
        chosen.append((grid[i[k]], grid[j[k]]))
        used += [a, b]
        if len(chosen) == PAIRS_PER_SET:
            break
    return chosen


def make_blocks(a, b, size=256):
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, : size // 2], img[:, size // 2 :] = a, b
    return img


def block_colors(img, size=256):
    # 只取各半邊內側，避開兩色交界 (模型在邊界會有過渡)
    m = size // 8
    return img[m:-m, m : size // 2 - m].mean((0, 1)), img[m:-m, size // 2 + m : -m].mean((0, 1))


def make_plate(a, b, digit, seed, size=512):
    """石原式檢查圖：回傳 (圖, 每個像素屬於 數字=1 / 背景=2 / 空白=0)。"""
    rng = np.random.RandomState(seed)
    mask_img = Image.new("L", (size, size), 0)
    try:
        font = ImageFont.truetype("arialbd.ttf", int(size * 0.62))
    except OSError:
        font = ImageFont.load_default(size=int(size * 0.62))
    ImageDraw.Draw(mask_img).text((size / 2, size / 2), digit, fill=255, font=font, anchor="mm")
    digit_mask = np.array(mask_img) > 0

    img = np.ones((size, size, 3), dtype=np.float32)
    label = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.mgrid[:size, :size]
    c, disc_r = size / 2, size * 0.47
    centers, radii = np.empty((0, 2)), np.empty(0)
    for _ in range(6000):
        r = rng.uniform(size * 0.008, size * 0.022)
        p = rng.uniform(c - disc_r, c + disc_r, 2)
        if np.hypot(*(p - c)) + r > disc_r:
            continue
        if len(radii) and (np.hypot(*(centers - p).T) < radii + r + 1.5).any():
            continue
        centers, radii = np.vstack([centers, p]), np.append(radii, r)
        is_digit = digit_mask[int(p[1]), int(p[0])]
        color = np.clip(np.asarray(a if is_digit else b) * rng.uniform(0.85, 1.0), 0, 1)
        dot = (xx - p[0]) ** 2 + (yy - p[1]) ** 2 <= r * r
        img[dot], label[dot] = color, 1 if is_digit else 2
    return img, label


def plate_colors(img, label):
    return img[label == 1].mean(0), img[label == 2].mean(0)


models = {}
for tag, path in CHECKPOINTS:
    m = LightUNetColorCorrector(use_cvd_condition=True).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    models[tag] = m.eval()

rows, plate_examples = [], []
summary = {}  # (set, kind) -> {"before": [...], tag: [...], "natural_"+tag: [...]}

for set_name, cvd_type, tritan_model in PAIR_SETS:
    cvd_simulation.set_tritan_model(tritan_model)
    idx = torch.tensor([CVD_TYPES.index(cvd_type)], device=device)
    pairs = find_pairs(cvd_type)
    print(f"\n[{set_name}] 找到 {len(pairs)} 組混淆色對")

    for p_i, (a, b) in enumerate(pairs):
        tests = [("block", make_blocks(a, b), None)]
        plate, label = make_plate(a, b, DIGITS[p_i % len(DIGITS)], seed=p_i)
        tests.append(("plate", plate, label))

        for kind, img, label in tests:
            sample = (lambda x: block_colors(x)) if kind == "block" else (lambda x, l=label: plate_colors(x, l))
            x = to_tensor(img)
            with torch.no_grad():
                before = sample(to_np(simulate_cvd(x, cvd_type)))
                corrected = {tag: model(x, idx).clamp(0, 1) for tag, model in models.items()}
                after = {tag: sample(to_np(simulate_cvd(c, cvd_type))) for tag, c in corrected.items()}
            orig = sample(img)
            row = {
                "set": set_name, "kind": kind, "pair": p_i,
                "color_a": tuple(np.round(np.asarray(a) * 255).astype(int)),
                "color_b": tuple(np.round(np.asarray(b) * 255).astype(int)),
                "dE_normal": de(*orig), "dE_sim_before": de(*before),
            }
            for tag in models:
                row[f"dE_sim_{tag}"] = de(*after[tag])
                corr_cols = sample(to_np(corrected[tag]))
                # 自然度代價：校正後顏色跟原本顏色差多少 (正常視覺，兩色平均)
                row[f"dE_natural_{tag}"] = (de(orig[0], corr_cols[0]) + de(orig[1], corr_cols[1])) / 2
            rows.append(row)
            s = summary.setdefault((set_name, kind), {})
            for key in ["dE_sim_before"] + [f"dE_sim_{t}" for t in models] + [f"dE_natural_{t}" for t in models]:
                s.setdefault(key, []).append(row[key])

            if kind == "plate" and p_i == 0:
                plate_examples.append((set_name, img, to_np(simulate_cvd(x, cvd_type)),
                                       to_np(simulate_cvd(corrected["V9"], cvd_type))))
cvd_simulation.set_tritan_model("machado2009")

# ---- 數字輸出 ----
fields = list(rows[0].keys())
with open("../confusion_pairs_results.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})

print("\n色弱模擬下兩色的 ΔE2000 (越大越分得出來)，各組 8 對的平均；自然度 = 校正後跟原色差多少 (越小越自然)")
print(f"{'組別':<18}{'測試':<7}{'校正前':>8}{'V9':>8}{'V8':>8}{'V9自然度':>10}{'V8自然度':>10}")
for (set_name, kind), s in summary.items():
    print(f"{set_name:<18}{kind:<7}{np.mean(s['dE_sim_before']):>8.2f}{np.mean(s['dE_sim_V9']):>8.2f}"
          f"{np.mean(s['dE_sim_V8']):>8.2f}{np.mean(s['dE_natural_V9']):>10.2f}{np.mean(s['dE_natural_V8']):>10.2f}")

# ---- 長條圖：每組 (色塊/檢查圖) 校正前 vs V9 vs V8 ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
names = [p[0] for p in PAIR_SETS if (p[0], "block") in summary]
x = np.arange(len(names))
bw = 0.26
for ax, kind, title in zip(axes, ["block", "plate"], ["Confusion-pair blocks", "Ishihara-style plates"]):
    for k, key in enumerate(["before", "V9", "V8"]):
        col = "dE_sim_before" if key == "before" else f"dE_sim_{key}"
        means = [np.mean(summary[(n, kind)][col]) for n in names]
        bars = ax.bar(x + (k - 1) * bw, means, bw - 0.03, color=COLORS[key],
                      label={"before": "Before correction", "V9": "V9 (main)", "V8": "V8 (control)"}[key])
        if key == "V9":
            ax.bar_label(bars, fmt="%.1f", fontsize=8, color="#333333", padding=2)
    ax.set_xticks(x, names, fontsize=9)
    ax.set_title(title, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5df", linewidth=0.8)
    ax.set_axisbelow(True)
axes[0].set_ylabel("ΔE2000 under CVD simulation (higher = easier to tell apart)", fontsize=9)
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle(f"Mean over {PAIRS_PER_SET} held-out confusion pairs per set", fontsize=10, x=0.01, ha="left", color="#555555")
fig.tight_layout()
fig.savefig("../quantify_pairs.png", dpi=150)

# ---- 檢查圖範例 ----
fig, axes = plt.subplots(len(plate_examples), 3, figsize=(9, 3 * len(plate_examples)))
for r, (set_name, orig, before, after) in enumerate(plate_examples):
    for c, (im, t) in enumerate([(orig, "Original"), (before, "CVD sim, before"), (after, "CVD sim, after V9")]):
        axes[r, c].imshow(im)
        axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
        axes[r, c].set_title(f"{set_name}: {t}" if c == 0 else t, fontsize=9, loc="left")
fig.tight_layout()
fig.savefig("../quantify_plates.png", dpi=110)
print("\n已存：../confusion_pairs_results.csv、../quantify_pairs.png、../quantify_plates.png")

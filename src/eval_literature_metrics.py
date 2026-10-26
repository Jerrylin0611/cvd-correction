"""
用「文獻上別人提出的」指標評估 V8 / V9，回應老師 10/8 建議第 2 點 (指標需要文獻驗證)。

背景：distinguish / palette 是我們自己定義、也拿來當訓練 loss 的指標，用它們評估等於
球員兼裁判。這裡改用 CVD 校正綜述 Zhu & Mao 2021 (The Visual Computer,
doi:10.1007/s00371-021-02240-0) 第 3.2 節整理的 dichromacy 校正常用量化指標：

對比 (都要經過色弱模擬)：
- GCD  global chromatic diversity (Ma et al. 2008; Chen et al. 2011)，越高越好
       GCD(i,j) = ||sim(c'_i) - sim(c'_j)||，i,j 取遍圖中的顏色對 (CIELAB)。量的是色弱者
       看到的「絕對」對比強度，不管跟原圖對不對得上。另外印出原圖正常視覺的 GCD 當上限參考，
       「補回 %」= (校正後 - 不校正) / (原圖 - 不校正)，代表色弱損失的對比補回幾成。
- GCE  global contrast error (Huang et al. 2007; Hassan 2019)
       GCE(i,j) = ( ||c_i - c_j|| - ||sim(c'_i) - sim(c'_j)|| )^2，i,j 取遍圖中的顏色對。
       c 是原圖顏色、c' 是校正後顏色，都在 CIELAB。越低越好。跟我們的 palette 一樣是「全域」對比。
- LCE  local contrast error (Machado & Oliveira 2010; Zhu et al. 2019; Wang et al. 2021)
       LCE(i) = 1/|Ω_i| Σ_{j∈Ω_i} ( (||sim(c'_i) - sim(c'_j)|| - ||c_i - c_j||) / 160 )^2，
       Ω_i 是像素 i 鄰域內隨機挑的像素。越低越好。跟我們的 distinguish 一樣是「局部」對比。
GCE / LCE 是「跟原圖的對比一致」的誤差，對比不足或過強都會扣分；GCD 只看夠不夠強。
自然度 (原圖 vs 校正後，不經模擬)：
- CD     chromatic difference，CIELAB 歐氏距離 (λ=1) 全圖平均，越低越好。
         跟 eval_metrics.py 的 PCDM 同一類，只是 PCDM 用 CIEDE2000。
- FSIMc  feature similarity，彩色版 (Zhang et al. 2011)，piq 套件實作，越高越好。
(SSIM 也是綜述列的自然度指標，已經在 eval_metrics.py 裡。)

綜述沒規定的實作細節 (我們自己的選擇，換了數字會變但趨勢不受影響)：
- GCD / GCE 的「顏色」：每張圖固定 seed 隨機抽 512 個像素，取所有像素對。
- LCE 的 Ω_i：每個像素在 15x15 視窗內固定 seed 隨機挑 8 個鄰居 (超出邊界的夾回邊界)。
- 圖片前處理跟 eval_metrics.py / _eval_kodak.py 一樣直接 resize 成 256x256。
綜述表 3~6 的數值是在他們自己的 6 張圖上算的，影像不同，數字不能直接對照；
這裡看的是「相對不校正改善多少」以及 V8/V9 的排序。

另外對每張圖算 distinguish/palette (自創) 跟 LCE/GCE (文獻) 的 Spearman 相關，
檢查自創指標的判斷跟文獻指標是否一致。

用法: python eval_literature_metrics.py   (Kodak 需先下載到 ../data/kodak/，見 _eval_kodak.py)
結果累加到 ../literature_metrics_summary.csv
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import piq
import torch
from PIL import Image
from scipy.stats import spearmanr

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import cvd_simulation
from cvd_simulation import CVD_TYPES, simulate_cvd, srgb_to_linear
from dataset import split_image_paths
from losses import distinguishability_loss, palette_distance_loss
from model import LightUNetColorCorrector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = [
    ("V8", "checkpoints_v8/model_epoch50.pt"),
    ("V9", "checkpoints_v9/model_epoch50.pt"),
]
DATASETS = [
    ("COCO test", "../data/val2017", "test"),
    ("Kodak", "../data/kodak", "all"),
]
IMAGE_SIZE = 256
BATCH_SIZE = 16
GCE_SAMPLES = 512
LCE_NEIGHBORS = 8
LCE_RADIUS = 7
LCE_NORM = 160.0
SEED = 0
SUMMARY_FILE = Path("../literature_metrics_summary.csv")

# sRGB(D65) -> XYZ
_RGB2XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])


def rgb_to_lab(img):
    """(B,3,H,W) sRGB 0~1 -> CIELAB (B,3,H,W)，D65。"""
    xyz = torch.einsum("ij,bjhw->bihw", _RGB2XYZ.to(img.device), srgb_to_linear(img.clamp(0, 1)))
    xyz = xyz / _WHITE.to(img.device).view(1, 3, 1, 1)
    d = 6 / 29
    f = torch.where(xyz > d ** 3, xyz.clamp(min=1e-12) ** (1 / 3), xyz / (3 * d ** 2) + 4 / 29)
    return torch.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def gcd(lab, idx):
    """每張圖一個值：抽樣像素的所有顏色對，CIELAB 距離平均。"""
    x = lab.flatten(2)[:, :, idx].transpose(1, 2)
    return torch.cdist(x, x).mean((1, 2))


def gce(orig_lab, sim_lab, idx):
    """每張圖一個值：抽樣像素的所有顏色對，(原圖距離 - 校正後模擬距離)^2 的平均。"""
    o = orig_lab.flatten(2)[:, :, idx].transpose(1, 2)  # (B, N, 3)
    s = sim_lab.flatten(2)[:, :, idx].transpose(1, 2)
    return ((torch.cdist(o, o) - torch.cdist(s, s)) ** 2).mean((1, 2))


def lce(orig_lab, sim_lab, offsets):
    """每張圖一個值：每個像素跟隨機鄰居的局部對比誤差，全圖平均。"""
    _, _, h, w = orig_lab.shape
    ys = torch.arange(h, device=orig_lab.device).view(h, 1)
    xs = torch.arange(w, device=orig_lab.device).view(1, w)
    total = 0.0
    for dy, dx in offsets:  # dy, dx: (H, W) 每個像素各自的隨機位移
        ny, nx = (ys + dy).clamp(0, h - 1), (xs + dx).clamp(0, w - 1)
        d_orig = (orig_lab - orig_lab[:, :, ny, nx]).norm(dim=1)
        d_sim = (sim_lab - sim_lab[:, :, ny, nx]).norm(dim=1)
        total = total + (((d_sim - d_orig) / LCE_NORM) ** 2).mean((1, 2))
    return total / len(offsets)


def load_model(path):
    model = LightUNetColorCorrector(use_cvd_condition=True).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_images(root, split):
    paths = split_image_paths(root, split=split, extensions={".jpg", ".jpeg", ".png"}, recursive=False)
    imgs = [
        np.asarray(Image.open(p).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255.0
        for p in paths
    ]
    return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)


def per_image_loss(fn, out, orig, idx):
    return [fn(out[k : k + 1], orig[k : k + 1], cvd_type_idx=idx[k : k + 1]).item() for k in range(len(out))]


def evaluate(models, images, cvd_type):
    """回傳 {"不校正" or tag: {指標: 每張圖的 np.array}}。"""
    gen = torch.Generator().manual_seed(SEED)
    gce_idx = torch.randperm(IMAGE_SIZE * IMAGE_SIZE, generator=gen)[:GCE_SAMPLES].to(device)
    offsets = []
    while len(offsets) < LCE_NEIGHBORS:
        dy, dx = torch.randint(-LCE_RADIUS, LCE_RADIUS + 1, (2, IMAGE_SIZE, IMAGE_SIZE), generator=gen)
        dx = torch.where((dy == 0) & (dx == 0), torch.ones_like(dx), dx)  # 不跟自己比
        offsets.append((dy.to(device), dx.to(device)))

    keys = ["不校正"] + list(models)
    res = {k: {m: [] for m in ("gcd", "gcd_orig", "gce", "lce", "cd", "fsimc", "distinguish", "palette")} for k in keys}
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            orig = images[start : start + BATCH_SIZE].to(device)
            idx = torch.full((orig.shape[0],), CVD_TYPES.index(cvd_type), device=device, dtype=torch.long)
            orig_lab = rgb_to_lab(orig)
            outs = {"不校正": orig}
            outs.update({tag: m(orig, idx).clamp(0, 1) for tag, m in models.items()})
            for k, out in outs.items():
                sim_lab = rgb_to_lab(simulate_cvd(out, cvd_type))
                r = res[k]
                r["gcd"] += gcd(sim_lab, gce_idx).tolist()
                r["gcd_orig"] += gcd(orig_lab, gce_idx).tolist()
                r["gce"] += gce(orig_lab, sim_lab, gce_idx).tolist()
                r["lce"] += lce(orig_lab, sim_lab, offsets).tolist()
                r["cd"] += (rgb_to_lab(out) - orig_lab).norm(dim=1).mean((1, 2)).tolist()
                r["fsimc"] += piq.fsim(orig, out, data_range=1.0, reduction="none", chromatic=True).tolist()
                r["distinguish"] += per_image_loss(distinguishability_loss, out, orig, idx)
                r["palette"] += per_image_loss(palette_distance_loss, out, orig, idx)
    return {k: {m: np.array(v) for m, v in r.items()} for k, r in res.items()}


models = {tag: load_model(path) for tag, path in CHECKPOINTS}
data = {name: load_images(root, split) for name, root, split in DATASETS}
settings = [(t, "-") for t in CVD_TYPES if t != "tritanopia"]
settings += [("tritanopia", m) for m in cvd_simulation.TRITAN_MODELS]

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
new_summary = not SUMMARY_FILE.exists()
with open(SUMMARY_FILE, "a", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    if new_summary:
        w.writerow(["timestamp", "dataset", "num_images", "cvd_type", "tritan_sim", "method",
                    "gcd", "gcd_orig", "gcd_recover_%", "gce", "lce", "cd", "fsimc",
                    "gce_improve_%", "lce_improve_%"])
    for cvd_type, sim_model in settings:
        if sim_model != "-":
            cvd_simulation.set_tritan_model(sim_model)
        print(f"\n[{cvd_type}" + (f" / {sim_model}]" if sim_model != "-" else "]"))
        for name, images in data.items():
            res = evaluate(models, images, cvd_type)
            g0, l0 = res["不校正"]["gce"].mean(), res["不校正"]["lce"].mean()
            d0, d_orig = res["不校正"]["gcd"].mean(), res["不校正"]["gcd_orig"].mean()
            for k, r in res.items():
                g, l = r["gce"].mean(), r["lce"].mean()
                gi, li = (g0 - g) / g0 * 100, (l0 - l) / l0 * 100
                d = r["gcd"].mean()
                di = (d - d0) / (d_orig - d0) * 100
                print(f"  {name:<10} {k:<4} GCD={d:6.2f}/{d_orig:6.2f} (補回{di:+6.1f}%)  GCE={g:9.2f} ({gi:+5.1f}%)  LCE={l:.5f} ({li:+5.1f}%)  "
                      f"CD={r['cd'].mean():6.2f}  FSIMc={r['fsimc'].mean():.4f}")
                w.writerow([timestamp, name, len(images), cvd_type, sim_model, k,
                            f"{d:.4f}", f"{d_orig:.4f}", f"{di:.2f}", f"{g:.4f}", f"{l:.6f}", f"{r['cd'].mean():.4f}", f"{r['fsimc'].mean():.6f}",
                            f"{gi:.2f}", f"{li:.2f}"])
            # 自創 vs 文獻指標的一致性：把不校正 + 各模型的每張圖都放在一起算 Spearman
            pool = lambda m: np.concatenate([res[k][m] for k in res])
            rho_l = spearmanr(pool("distinguish"), pool("lce"))[0]
            rho_g = spearmanr(pool("palette"), pool("gce"))[0]
            print(f"  {name:<10} Spearman: distinguish~LCE={rho_l:.3f}  palette~GCE={rho_g:.3f}")
cvd_simulation.set_tritan_model("machado2009")
print(f"\n平均值已累加到：{SUMMARY_FILE}")

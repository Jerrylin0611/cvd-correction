"""
跨資料集泛化：把 V9 (主模型) / V8 (對照) 直接拿到 Kodak 24 張上推論 (不重訓)，
看在 COCO 上學到的校正效果能不能搬到完全沒看過的資料集。

背景 (老師 10/8 建議第 1 點：交叉比對泛化能力)：
- V8 已經驗證過 COCO 內部 train/test 切分的泛化 (見 _verify_v8_holdout.py)，
  但 test 跟 train 來自同一個資料集，分布很接近。Kodak 是另一組相機、另一種
  題材 (高解析、色彩飽和的風景/人像)，可以看「換一個資料分布」還有沒有效。
- 保真度指標 (SSIM/LPIPS/PCDM…) 直接用 eval_metrics.py 跑，tag 用 kodak_v9 / kodak_v8：
    python eval_metrics.py --checkpoint checkpoints_v9/model_epoch50.pt --tag kodak_v9 --data_dir ../data/kodak --split all
  這支腳本只算可辨識度 (distinguish/palette)。

為了讓 Kodak 跟 COCO 的數字可以並排比，兩邊都用同一種確定性前處理
(直接 resize 成 256x256，跟 eval_metrics.py 一致)，不用 ColorImageFolder 的隨機裁切；
所以這裡的 COCO test 數字跟 _verify_v8_holdout.py / _verify_brettel_tritan.py 會有小差異。
tritan 兩種模擬 (Machado 2009 / Brettel 1997) 都報，理由見 _verify_brettel_tritan.py。

另外算每張原圖的 colorfulness (Hasler & Süsstrunk 2003)，當作「這個資料集的圖本來
有多鮮豔」的參考：越鮮豔，色弱看起來損失越多、模型也需要改得越用力，保真度指標
自然會比較差，不全是泛化問題。

Kodak 需要自己下載到 ../data/kodak/ (不進版控)：
    https://r0k.us/graphics/kodak/kodak/kodim01.png ~ kodim24.png

用法: python _eval_kodak.py
"""

import sys

import numpy as np
import torch
from PIL import Image

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import cvd_simulation
from cvd_simulation import CVD_TYPES
from dataset import split_image_paths
from losses import distinguishability_loss, palette_distance_loss
from model import LightUNetColorCorrector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = [
    ("V8(對照)", "checkpoints_v8/model_epoch50.pt"),
    ("V9(主模型)", "checkpoints_v9/model_epoch50.pt"),
]
DATASETS = [
    ("COCO test", "../data/val2017", "test"),
    ("Kodak", "../data/kodak", "all"),
]
IMAGE_SIZE = 256
BATCH_SIZE = 32


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
    return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)  # (N, 3, H, W)，0~1，先放 CPU


def colorfulness(images):
    """Hasler & Süsstrunk 2003 的 colorfulness，0~255 尺度，每張圖一個值。"""
    rgb = images * 255.0
    rg = rgb[:, 0] - rgb[:, 1]
    yb = 0.5 * (rgb[:, 0] + rgb[:, 1]) - rgb[:, 2]
    std = torch.sqrt(rg.flatten(1).std(1) ** 2 + yb.flatten(1).std(1) ** 2)
    mean = torch.sqrt(rg.flatten(1).mean(1) ** 2 + yb.flatten(1).mean(1) ** 2)
    return (std + 0.3 * mean).numpy()


def mean_losses(models, images, cvd_type):
    """回傳 {"不校正" or tag: (distinguish, palette)}，按張數加權平均 (兩個 loss 都是逐像素平均，跟 batch 組成無關)。"""
    sums = {k: np.zeros(2) for k in ["不校正"] + list(models)}
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            orig = images[start : start + BATCH_SIZE].to(device)
            idx = torch.full((orig.shape[0],), CVD_TYPES.index(cvd_type), device=device, dtype=torch.long)
            outs = {"不校正": orig}
            outs.update({tag: m(orig, idx).clamp(0, 1) for tag, m in models.items()})
            for k, out in outs.items():
                d = distinguishability_loss(out, orig, cvd_type_idx=idx).item()
                p = palette_distance_loss(out, orig, cvd_type_idx=idx).item()
                sums[k] += np.array([d, p]) * orig.shape[0]
    return {k: v / len(images) for k, v in sums.items()}


models = {tag: load_model(path) for tag, path in CHECKPOINTS}
data = {name: load_images(root, split) for name, root, split in DATASETS}

print("原圖 colorfulness (Hasler & Süsstrunk 2003，越高越鮮豔)")
for name, images in data.items():
    c = colorfulness(images)
    print(f"  {name:<10} n={len(images):<4} {c.mean():.1f} ± {c.std():.1f}")

# (色弱類型, tritan 模擬模型)；protan/deuter 不受 tritan 模擬切換影響
settings = [(t, None) for t in CVD_TYPES if t != "tritanopia"]
settings += [("tritanopia", m) for m in cvd_simulation.TRITAN_MODELS]

print("\n可辨識度 (distinguish / palette loss，越低越好；括號是相對不校正的改善 %)")
for cvd_type, sim_model in settings:
    if sim_model:
        cvd_simulation.set_tritan_model(sim_model)
    label = cvd_type + (f" [{sim_model}]" if sim_model else "")
    print(f"\n[{label}]")
    for name, images in data.items():
        res = mean_losses(models, images, cvd_type)
        d0, p0 = res["不校正"]
        line = f"  {name:<10} 不校正 d={d0:.5f} p={p0:.5f}"
        for tag in models:
            d1, p1 = res[tag]
            line += f" | {tag} d={(d0 - d1) / d0 * 100:+.1f}% p={(p0 - p1) / p0 * 100:+.1f}%"
        print(line)
cvd_simulation.set_tritan_model("machado2009")

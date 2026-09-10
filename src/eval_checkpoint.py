"""
評估腳本：把一個 checkpoint 的關鍵指標算出來，並「累加」記錄到 results.csv，
方便每次調參數重跑後，直接跟之前所有實驗版本一起比較，不用每次重算舊的。

量測兩種情境：
1. 合成的純紅/純綠色塊測試 (deuteranopia)：抓「大面積純色、無紋理」這種
   COCO 照片幾乎沒有的分布死角，看模型會不會失敗 (例如把紅色整塊塗成綠色)。
2. 真實照片批次 (COCO val2017，固定 random seed 方便跨版本比較)：
   看一般拍照場景下的 distinguishability / palette loss。

用法範例:
    python eval_checkpoint.py --checkpoint checkpoints_v3/model_epoch50.pt \
        --tag v3 --notes "palette+bias loss, synthetic_ratio=0.15"
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from cvd_simulation import CVD_TYPES, simulate_cvd
from dataset import ColorImageFolder
from losses import distinguishability_loss, palette_distance_loss
from model import LightUNetColorCorrector

RESULTS_COLUMNS = [
    "timestamp", "tag", "checkpoint", "notes",
    "pure_color_diff_before", "pure_color_diff_after",
    "real_distinguish_before", "real_distinguish_after",
    "real_palette_before", "real_palette_after",
    "real_avg_pixel_change",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True, help="這次實驗的簡短代號，例如 v3, v4")
    parser.add_argument("--notes", type=str, default="", help="這次實驗的設定描述，例如 loss 權重/資料比例")
    parser.add_argument("--data_dir", type=str, default="../data/val2017")
    parser.add_argument("--cvd_type", type=str, default="deuteranopia", choices=CVD_TYPES)
    parser.add_argument("--seed", type=int, default=0, help="固定 seed 抽同一批照片，確保跨版本結果可比較")
    parser.add_argument("--results_file", type=str, default="../results.csv")
    return parser.parse_args()


def sample_halves(t, size):
    left = t[0, :, :, : size // 2].mean(dim=[1, 2]).cpu().numpy() * 255
    right = t[0, :, :, size // 2 :].mean(dim=[1, 2]).cpu().numpy() * 255
    return left, right


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LightUNetColorCorrector(use_cvd_condition=True).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    cvd_idx_scalar = torch.tensor([CVD_TYPES.index(args.cvd_type)], device=device)

    # --- 1. 純紅/純綠合成色塊測試 ---
    size = 256
    raw = np.zeros((size, size, 3), dtype=np.uint8)
    raw[:, : size // 2] = [255, 0, 0]
    raw[:, size // 2 :] = [0, 255, 0]
    img_tensor = torch.from_numpy(raw).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    before_sim = simulate_cvd(img_tensor, args.cvd_type)
    l, r = sample_halves(before_sim, size)
    pure_diff_before = float(np.linalg.norm(l - r))

    with torch.no_grad():
        corrected = model(img_tensor, cvd_idx_scalar)
        after_sim = simulate_cvd(corrected, args.cvd_type)
    l, r = sample_halves(after_sim, size)
    pure_diff_after = float(np.linalg.norm(l - r))

    # --- 2. 真實照片批次 (固定 seed) ---
    torch.manual_seed(args.seed)
    dataset = ColorImageFolder(args.data_dir, image_size=256)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    images, cvd_type_idx = next(iter(loader))
    images, cvd_type_idx = images.to(device), cvd_type_idx.to(device)

    real_dist_before = distinguishability_loss(images, images, CVD_TYPES).item()
    real_pal_before = palette_distance_loss(images, images, CVD_TYPES).item()

    with torch.no_grad():
        corrected_real = model(images, cvd_type_idx)
    real_dist_after = distinguishability_loss(corrected_real, images, CVD_TYPES).item()
    real_pal_after = palette_distance_loss(corrected_real, images, CVD_TYPES).item()
    avg_pixel_change = (corrected_real - images).abs().mean().item()

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "checkpoint": args.checkpoint,
        "notes": args.notes,
        "pure_color_diff_before": round(pure_diff_before, 2),
        "pure_color_diff_after": round(pure_diff_after, 2),
        "real_distinguish_before": round(real_dist_before, 5),
        "real_distinguish_after": round(real_dist_after, 5),
        "real_palette_before": round(real_pal_before, 5),
        "real_palette_after": round(real_pal_after, 5),
        "real_avg_pixel_change": round(avg_pixel_change, 5),
    }

    results_path = Path(args.results_file)
    file_exists = results_path.exists()
    with open(results_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[{args.tag}] 純色測試色差: {pure_diff_before:.1f} -> {pure_diff_after:.1f}")
    print(
        f"[{args.tag}] 真實照片 distinguish: {real_dist_before:.5f} -> {real_dist_after:.5f}  "
        f"palette: {real_pal_before:.5f} -> {real_pal_after:.5f}"
    )
    print(f"已記錄到 {results_path.resolve()}")


if __name__ == "__main__":
    main()

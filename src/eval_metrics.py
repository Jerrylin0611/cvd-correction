"""
影像品質指標評估：把「原圖」跟「校正後圖片」比較，算 SSIM / MS-SSIM / CW-SSIM /
LPIPS / PCDM，方便跟其他色弱校正方法（論文、GAN 類方法）在同一套指標下比較。

這些指標量的是「校正後有多忠於原圖」（結構、感知、色差），不是「色弱者有沒有比較
容易分辨」。後者請看 eval_checkpoint.py 的 distinguishability / palette 指標。

指標定義：
- SSIM     ：skimage structural_similarity，RGB 三通道平均，data_range=1。越高越好。
- MS-SSIM  ：pytorch_msssim，RGB。越高越好。
- CW-SSIM  ：灰階 (0~255)、複數 steerable pyramid，每個子頻帶用 7x7 局部視窗算
             (Sampat et al. 2009，K=0.01)，再對所有子頻帶/位置取平均。越高越好。
- LPIPS    ：AlexNet 版。越低越好。
- PCDM     ：CIEDE2000 色差的全圖平均（跟學弟的 test_pcdm.py 同定義）。越低越好。

用法範例（三種色弱類型都算，最常用）:
    python eval_metrics.py --checkpoint checkpoints_v7/model_epoch50.pt --tag v7

只測 deuteranopia、先抽 100 張試跑:
    python eval_metrics.py --checkpoint checkpoints_v7/model_epoch50.pt --tag v7 \
        --cvd_type deuteranopia --limit 100

每張圖的結果存在 ../metrics_results/，各版本的平均值累加記錄到 ../metrics_summary.csv。
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import lpips
import numpy as np
import pyrtools as pt
import torch
from PIL import Image
from pytorch_msssim import ms_ssim
from scipy.ndimage import uniform_filter
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from cvd_simulation import CVD_TYPES
from dataset import DEFAULT_SPLIT_SEED, DEFAULT_TEST_RATIO, split_image_paths
from model import LightUNetColorCorrector

METRICS = ["ssim", "ms_ssim", "cw_ssim", "lpips", "pcdm"]
SUMMARY_COLUMNS = (
    ["timestamp", "tag", "checkpoint", "cvd_type", "split", "num_images", "image_size"]
    + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True, help="這次評估的代號，例如 v7")
    parser.add_argument("--cvd_type", type=str, default="all", choices=["all"] + CVD_TYPES)
    parser.add_argument("--data_dir", type=str, default="../data/val2017")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--split", type=str, default="test", choices=["all", "train", "test"],
        help="預設只評估 held-out 測試集 (跟訓練時保留的那份一致)，避免拿模型訓練時看過的圖打分數；"
             "只有想重現舊版『全部都算』的行為時才用 all",
    )
    parser.add_argument("--test_ratio", type=float, default=DEFAULT_TEST_RATIO, help="要跟 train.py 的設定一致")
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED, help="要跟 train.py 的設定一致")
    parser.add_argument("--limit", type=int, default=0, help="只測前 N 張，0 = 全部")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="../metrics_results")
    parser.add_argument("--summary_file", type=str, default="../metrics_summary.csv")
    return parser.parse_args()


def cw_ssim_from_pyramids(pyr1, pyr2, K=0.01, win=7):
    """兩個複數 steerable pyramid 之間的 CW-SSIM（局部視窗版，跨子頻帶平均）。"""
    scores = []
    for key, c1 in pyr1.pyr_coeffs.items():
        if not isinstance(key, tuple):  # 跳過 residual highpass / lowpass
            continue
        c2 = pyr2.pyr_coeffs[key]
        cross = c1 * np.conj(c2)
        num = np.hypot(
            uniform_filter(cross.real, win, mode="reflect"),
            uniform_filter(cross.imag, win, mode="reflect"),
        )
        den = uniform_filter(np.abs(c1) ** 2, win, mode="reflect") + uniform_filter(
            np.abs(c2) ** 2, win, mode="reflect"
        )
        scores.append(np.mean((2 * num + K) / (den + K)))
    return float(np.mean(scores))


def to_pyramid(gray255):
    return pt.pyramids.SteerablePyramidFreq(gray255, height="auto", order=3, is_complex=True)


def rgb_to_gray255(img01):
    return np.dot(img01, [0.299, 0.587, 0.114]) * 255.0


def load_model(path, device):
    model = LightUNetColorCorrector(use_cvd_condition=True).to(device)
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    return model


def load_batch(paths, size):
    imgs = [
        np.asarray(Image.open(p).convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
        for p in paths
    ]
    return np.stack(imgs)  # (B, H, W, 3)，0~1


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvd_types = CVD_TYPES if args.cvd_type == "all" else [args.cvd_type]

    files = split_image_paths(
        args.data_dir, split=args.split, test_ratio=args.test_ratio, split_seed=args.split_seed,
        extensions={".jpg", ".jpeg", ".png"}, recursive=False,
    )
    if args.limit:
        files = files[: args.limit]
    print(
        f"裝置：{device}｜split={args.split}｜圖片數：{len(files)}｜"
        f"色弱類型：{cvd_types}｜size={args.image_size}"
    )

    model = load_model(args.checkpoint, device)
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    rows = {t: [] for t in cvd_types}  # 每種色弱類型 -> 每張圖的指標 dict

    with torch.no_grad():
        for start in range(0, len(files), args.batch_size):
            batch_paths = files[start : start + args.batch_size]
            orig_np = load_batch(batch_paths, args.image_size)
            orig = torch.from_numpy(orig_np).permute(0, 3, 1, 2).to(device)
            orig_lab = [rgb2lab(im) for im in orig_np]
            orig_pyr = [to_pyramid(rgb_to_gray255(im)) for im in orig_np]  # 各色弱類型共用

            for cvd in cvd_types:
                idx = torch.full((len(batch_paths),), CVD_TYPES.index(cvd), device=device)
                corr = model(orig, idx).clamp(0, 1)
                corr_np = corr.permute(0, 2, 3, 1).cpu().numpy()

                ms = ms_ssim(orig, corr, data_range=1.0, size_average=False).cpu().numpy()
                lp = lpips_fn(orig * 2 - 1, corr * 2 - 1).flatten().cpu().numpy()

                for i, path in enumerate(batch_paths):
                    rows[cvd].append({
                        "image": path.name,
                        "ssim": structural_similarity(
                            orig_np[i], corr_np[i], data_range=1.0, channel_axis=2
                        ),
                        "ms_ssim": float(ms[i]),
                        "cw_ssim": cw_ssim_from_pyramids(
                            orig_pyr[i], to_pyramid(rgb_to_gray255(corr_np[i]))
                        ),
                        "lpips": float(lp[i]),
                        "pcdm": float(np.mean(deltaE_ciede2000(orig_lab[i], rgb2lab(corr_np[i])))),
                    })
            print(f"  {min(start + args.batch_size, len(files))}/{len(files)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_file)
    new_summary = not summary_path.exists()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(summary_path, "a", newline="", encoding="utf-8-sig") as sf:
        sw = csv.writer(sf)
        if new_summary:
            sw.writerow(SUMMARY_COLUMNS)
        print()
        for cvd in cvd_types:
            per_image = rows[cvd]
            with open(out_dir / f"metrics_{args.tag}_{cvd}.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["image"] + METRICS)
                for r in per_image:
                    w.writerow([r["image"]] + [f"{r[m]:.6f}" for m in METRICS])

            stats = {m: np.array([r[m] for r in per_image], dtype=np.float64) for m in METRICS}
            sw.writerow(
                [timestamp, args.tag, args.checkpoint, cvd, args.split, len(per_image), args.image_size]
                + [f"{getattr(stats[m], s)():.6f}" for m in METRICS for s in ("mean", "std")]
            )
            print(f"[{cvd}] " + "  ".join(f"{m}={stats[m].mean():.4f}±{stats[m].std():.4f}" for m in METRICS))

    print(f"\n每張圖結果：{out_dir}\n平均值已累加到：{summary_path}")


if __name__ == "__main__":
    main()

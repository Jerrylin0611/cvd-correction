"""
訓練腳本。

用法範例:
    python train.py --data_dir ./data/train_images --epochs 30 --batch_size 8

--data_dir 指向一個資料夾，裡面放任意一般彩色圖片即可 (不需要標註)，
細節說明見 dataset.py 開頭的註解。
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

# Windows 終端機預設編碼常常不是 UTF-8，會讓中文註解/print 出現亂碼，這裡強制改成 UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dataset import DEFAULT_SPLIT_SEED, DEFAULT_TEST_RATIO, ColorImageFolder
from losses import ColorCorrectionLoss
from model import LightUNetColorCorrector
from synthetic_colors import ConfusionPairBlocks, SyntheticColorBlocks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="訓練圖片所在資料夾")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--test_ratio", type=float, default=DEFAULT_TEST_RATIO,
        help="保留給 eval_metrics.py/eval_checkpoint.py 當 held-out 測試集的比例，"
             "訓練時不會用到這部分圖片；要跟 eval 腳本的 --test_ratio 一致才不會切出不同的切法",
    )
    parser.add_argument(
        "--split_seed", type=int, default=DEFAULT_SPLIT_SEED,
        help="train/test 切分用的固定 seed，要跟 eval 腳本的 --split_seed 一致",
    )
    parser.add_argument("--w_natural", type=float, default=0.1, help="自然度 loss 權重")
    parser.add_argument("--w_structure", type=float, default=0.1, help="結構(SSIM) loss 權重")
    parser.add_argument("--w_distinguish", type=float, default=10.0, help="可辨識度(邊緣型) loss 權重")
    parser.add_argument("--w_palette", type=float, default=10.0, help="可辨識度(調色盤型) loss 權重，補強大面積色塊")
    parser.add_argument("--w_bias", type=float, default=5.0, help="整體色偏懲罰項權重 (避免整張圖套濾鏡)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every", type=int, default=5, help="每幾個 epoch 存一次checkpoint")
    parser.add_argument(
        "--synthetic_ratio", type=float, default=0.15,
        help="混入訓練集的合成純色色塊圖比例 (相對於真實照片張數)，"
             "用來補強 COCO 照片幾乎沒有的大面積純色/無紋理分布，見 synthetic_colors.py",
    )
    parser.add_argument(
        "--targeted_ratio", type=float, default=0.0,
        help="混入訓練集的『針對性混淆色對』合成圖比例 (相對於真實照片張數)，"
             "只生成各色弱類型真正會混淆的顏色對 (見 synthetic_colors.ConfusionPairBlocks)，"
             "資料量小但每筆都打在死角上，用來取代/補強 --synthetic_ratio 那種隨機版本",
    )
    parser.add_argument(
        "--targeted_cvd_type", type=str, default=None, choices=[None, "protanopia", "deuteranopia", "tritanopia"],
        help="搭配 --targeted_ratio 使用：只指定生成某一種色弱類型的混淆對色塊 "
             "(預設 None = 三種類型隨機混合，會連帶改變另外兩種類型的訓練資料分布)，"
             "用來做『只補強單一類型、其餘類型訓練分布不變』的乾淨對照實驗",
    )
    parser.add_argument(
        "--init_checkpoint", type=str, default=None,
        help="從既有 checkpoint 載入權重當起點做 fine-tune (而非從頭訓練)，"
             "例如接著 V2 的權重只針對某個弱點小修時使用；搭配較小的 --lr",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    dataset = ColorImageFolder(
        args.data_dir, image_size=args.image_size, split="train",
        test_ratio=args.test_ratio, split_seed=args.split_seed,
    )
    all_datasets = [dataset]
    dataset_desc = [f"{len(dataset)} (真實照片，已保留 {args.test_ratio:.0%} 當 held-out 測試集)"]

    n_synthetic = int(len(dataset) * args.synthetic_ratio)
    if n_synthetic > 0:
        all_datasets.append(SyntheticColorBlocks(length=n_synthetic, image_size=args.image_size))
        dataset_desc.append(f"{n_synthetic} (隨機合成純色色塊)")

    n_targeted = int(len(dataset) * args.targeted_ratio)
    if n_targeted > 0:
        all_datasets.append(
            ConfusionPairBlocks(length=n_targeted, image_size=args.image_size, cvd_type=args.targeted_cvd_type)
        )
        type_desc = args.targeted_cvd_type or "三類型隨機混合"
        dataset_desc.append(f"{n_targeted} (針對性混淆色對，{type_desc})")

    combined_dataset = ConcatDataset(all_datasets) if len(all_datasets) > 1 else dataset
    dataloader = DataLoader(
        combined_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    print(f"訓練圖片數量: " + " + ".join(dataset_desc))

    model = LightUNetColorCorrector(use_cvd_condition=True).to(device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        print(f"已從 {args.init_checkpoint} 載入權重，接續 fine-tune")
    criterion = ColorCorrectionLoss(
        w_natural=args.w_natural,
        w_structure=args.w_structure,
        w_distinguish=args.w_distinguish,
        w_palette=args.w_palette,
        w_bias=args.w_bias,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_stats = {
            "natural": 0.0, "structure": 0.0, "distinguish": 0.0,
            "palette": 0.0, "bias": 0.0, "total": 0.0,
        }

        for images, cvd_type_idx in dataloader:
            images = images.to(device)
            cvd_type_idx = cvd_type_idx.to(device)

            corrected = model(images, cvd_type_idx)

            # 用每張圖各自被條件輸入指定的色弱類型去算 distinguish/palette loss，
            # 而不是不管條件輸入、每張圖都跟三種類型的模擬結果比對——
            # 後者會讓模型學到「不管輸入哪種類型都做同一套通用校正」，
            # 條件輸入等於形同虛設 (webcam demo 實測：切到 tritanopia 模式，
            # 校正後紅色卻被推向綠色，就是這個問題)
            total_loss, stats = criterion(corrected, images, cvd_type_idx=cvd_type_idx)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            for k in epoch_stats:
                epoch_stats[k] += stats[k]

        n_batches = len(dataloader)
        avg_stats = {k: v / n_batches for k, v in epoch_stats.items()}
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"total={avg_stats['total']:.4f} "
            f"natural={avg_stats['natural']:.4f} "
            f"structure={avg_stats['structure']:.4f} "
            f"distinguish={avg_stats['distinguish']:.4f} "
            f"palette={avg_stats['palette']:.4f} "
            f"bias={avg_stats['bias']:.4f}"
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = checkpoint_dir / f"model_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"已儲存 checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

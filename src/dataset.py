"""
訓練資料集。

重點：這個訓練是「自監督式」的，不需要色弱標註資料！
只要準備一堆一般的彩色照片 (風景、物品、隨便一個公開圖片資料集都可以)，
訓練時網路會自己嘗試校正，loss 再用 CVD 模擬去檢查校正得好不好 (見 losses.py)。
所以這裡的 Dataset 只需要讀圖片、做基本的 resize/裁切，不用管標籤。
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from cvd_simulation import CVD_TYPES

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# train.py 跟 eval_metrics.py/eval_checkpoint.py 都要用同一組預設值，
# 這樣兩邊各自呼叫 split_image_paths 才會切出「不重疊」的 train/test。
# 不要在個別腳本裡各自硬寫一份，否則兩邊改了其中一個就會悄悄重疊。
DEFAULT_TEST_RATIO = 0.1
DEFAULT_SPLIT_SEED = 42


def split_image_paths(
    root_dir: str,
    split: str = "all",
    test_ratio: float = DEFAULT_TEST_RATIO,
    split_seed: int = DEFAULT_SPLIT_SEED,
    extensions=IMG_EXTENSIONS,
    recursive: bool = True,
) -> list:
    """
    把資料夾底下的圖片路徑切成 train/test，訓練跟評估腳本共用，確保兩邊用同一個
    seed + test_ratio 算出來的切法一致、不會重疊 (不用實際搬動檔案)。

    split="all"：不切，回傳全部 (舊行為，向後相容)。
    split="train"/"test"：用 split_seed 固定 shuffle 後，前 test_ratio 比例當 test，
    其餘當 train。同一個資料夾、同樣的 test_ratio/split_seed，多次呼叫結果都一樣。
    """
    root = Path(root_dir)
    it = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(p for p in it if p.suffix.lower() in extensions)
    if not paths:
        raise ValueError(f"在 {root_dir} 底下沒有找到任何圖片，請確認路徑是否正確")
    if split == "all":
        return paths
    if split not in ("train", "test"):
        raise ValueError(f"split 必須是 all/train/test，收到 {split!r}")

    shuffled = paths[:]
    random.Random(split_seed).shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_ratio))
    return shuffled[:n_test] if split == "test" else shuffled[n_test:]


class ColorImageFolder(Dataset):
    """讀取資料夾下的圖片 (可指定 train/test split)，隨機裁切成固定大小，並隨機指定一個色弱類型當訓練目標。"""

    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
        split: str = "all",
        test_ratio: float = DEFAULT_TEST_RATIO,
        split_seed: int = DEFAULT_SPLIT_SEED,
    ):
        self.paths = split_image_paths(root_dir, split, test_ratio, split_seed)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # 轉成 [0,1] 範圍的 tensor
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img_tensor = self.transform(img)

        # 每次隨機挑一種色弱類型當這個 sample 的訓練目標，
        # 讓同一個模型在訓練過程中把三種類型都學過一遍
        cvd_type_idx = random.randrange(len(CVD_TYPES))

        return img_tensor, torch.tensor(cvd_type_idx, dtype=torch.long)

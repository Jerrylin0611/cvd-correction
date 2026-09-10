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


class ColorImageFolder(Dataset):
    """遞迴讀取資料夾下所有圖片，隨機裁切成固定大小，並隨機指定一個色弱類型當訓練目標。"""

    def __init__(self, root_dir: str, image_size: int = 256):
        self.paths = [p for p in Path(root_dir).rglob("*") if p.suffix.lower() in IMG_EXTENSIONS]
        if not self.paths:
            raise ValueError(f"在 {root_dir} 底下沒有找到任何圖片，請確認路徑是否正確")

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

"""把幾張範例圖存成 PNG，方便直接用檔案總管/VSCode 打開用眼睛檢查效果。"""

import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from cvd_simulation import CVD_TYPES, simulate_cvd
from dataset import ColorImageFolder
from model import LightUNetColorCorrector

checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_exp1/model_epoch15.pt"
out_prefix = sys.argv[2] if len(sys.argv) > 2 else "example"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = ColorImageFolder("../data/val2017", image_size=256)
loader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=0)

model = LightUNetColorCorrector(use_cvd_condition=True).to(device)
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

images, _ = next(iter(loader))
images = images.to(device)

cvd_type = "deuteranopia"  # 最常見的色弱類型，用這個當展示範例
cvd_idx = CVD_TYPES.index(cvd_type)
cvd_type_idx = torch.full((images.shape[0],), cvd_idx, device=device, dtype=torch.long)

with torch.no_grad():
    corrected = model(images, cvd_type_idx)
    before_sim = simulate_cvd(images, cvd_type)
    after_sim = simulate_cvd(corrected, cvd_type)


def to_numpy(t):
    return (t.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


orig_np = to_numpy(images)
corrected_np = to_numpy(corrected)
before_np = to_numpy(before_sim)
after_np = to_numpy(after_sim)

rows = []
for i in range(images.shape[0]):
    row = np.hstack([orig_np[i], corrected_np[i], before_np[i], after_np[i]])
    rows.append(row)
grid = np.vstack(rows)

Image.fromarray(grid).save(f"{out_prefix}.png")
print(f"已儲存: {out_prefix}.png")
print("每一列由左到右: 原圖 / 校正後圖片(一般人視角) / 校正前的色弱模擬 / 校正後的色弱模擬")

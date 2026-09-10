"""驗證訓練出來的模型是否真的有在做顏色校正，而不是學到「輸出=輸入」的偷懶解。"""

import torch
from torch.utils.data import DataLoader

from cvd_simulation import CVD_TYPES, simulate_cvd
from dataset import ColorImageFolder
from model import LightUNetColorCorrector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ColorImageFolder("../data/val2017", image_size=256)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

model = LightUNetColorCorrector(use_cvd_condition=True).to(device)
model.load_state_dict(torch.load("checkpoints/model_epoch50.pt", map_location=device))
model.eval()

images, cvd_type_idx = next(iter(loader))
images, cvd_type_idx = images.to(device), cvd_type_idx.to(device)

with torch.no_grad():
    corrected = model(images, cvd_type_idx)

diff = (corrected - images).abs()
print(f"平均每個 pixel 校正量 (0~1 範圍): mean={diff.mean().item():.6f}, max={diff.max().item():.6f}")
print(f"有超過 0.01 校正量的 pixel 比例: {(diff > 0.01).float().mean().item()*100:.2f}%")
print(f"有超過 0.05 校正量的 pixel 比例: {(diff > 0.05).float().mean().item()*100:.2f}%")

# 對照組：完全不校正 (corrected = images)，理論上的 distinguish loss 應該比訓練後的模型差 (數值更高)
from losses import distinguishability_loss
no_correction_loss = distinguishability_loss(images, images, CVD_TYPES)
model_loss = distinguishability_loss(corrected, images, CVD_TYPES)
print(f"完全不校正的 distinguish loss: {no_correction_loss.item():.6f}")
print(f"模型校正後的 distinguish loss:   {model_loss.item():.6f}")

"""
比較 V7 / V8 / V9 / V10 在 held-out test split (500 張) 上的可辨識度
(distinguish/palette loss)，驗證「補強 tritanopia 合成混淆色塊資料」這個方向
到底有沒有用。

背景：
- V8：跟 V7 同設定，只是只用 train split 訓練 (held-out 對照組，無合成資料)。
- V9：V8 + --targeted_ratio 0.15 (三種色弱類型隨機混合的合成混淆色塊，
  用修正後的 tritan 混淆對)。結果：tritan 保真度指標大幅改善，但代價是
  三種類型的可辨識度全面變弱 —— 懷疑是合成資料稀釋/改變了三種類型的訓練分布。
- V10：V8 + --targeted_ratio 0.15 --targeted_cvd_type tritanopia (只生成
  tritan 的合成混淆色塊，不動 protan/deuter 的訓練分布)，驗證能否隔離掉
  V9 的副作用、只改善 tritanopia。

結論 (2026-09-23)：
- V10 的 protan/deuter 可辨識度改善幅度幾乎跟 V8 一樣 (差距 <1 個百分點)，
  證實「V9 拖累 protan/deuter」確實是合成資料改變訓練分布造成的，
  只補強單一類型可以避免波及其他類型。
- 但 V10 自己的 tritanopia 可辨識度反而比 V8 更差 (distinguish +61.6% vs
  V8 +70.1%，palette +73.5% vs V8 +83.1%)，跟 V9 一樣的方向。也就是說
  「加入合成混淆色塊」這個做法本身 (不管是否跟其他類型混在一起)，
  在 tritanopia 上都是讓校正變得更保守 (跟原圖更像、保真度指標變好)，
  但可辨識度變差 (校正力道變弱)，不是我們要的效果。
- 建議：合成色塊補強這條路目前看起來走不通，下次應該改查「COCO 訓練資料裡
  tritan 相關色相 (藍/綠/黃/粉) 的圖片是否真的比較少」，或考慮調高
  w_distinguish/w_palette 的權重、而不是加合成資料。

用法: python _verify_v10_targeted.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from cvd_simulation import CVD_TYPES
from dataset import ColorImageFolder
from losses import distinguishability_loss, palette_distance_loss
from model import LightUNetColorCorrector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = [
    ("V7(全資料)", "checkpoints_v7/model_epoch50.pt"),
    ("V8(held-out)", "checkpoints_v8/model_epoch50.pt"),
    ("V9(+targeted 三類型)", "checkpoints_v9/model_epoch50.pt"),
    ("V10(+targeted 僅tritan)", "checkpoints_v10/model_epoch50.pt"),
]


def load_model(path):
    model = LightUNetColorCorrector(use_cvd_condition=True).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def weighted_avg(values, weights):
    values, weights = np.array(values), np.array(weights)
    return float((values * weights).sum() / weights.sum())


dataset = ColorImageFolder("../data/val2017", image_size=256, split="test")
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
all_images = [batch.to(device) for batch, _ in loader]
models = {tag: load_model(path) for tag, path in CHECKPOINTS}

print("可辨識度 (distinguish/palette loss，越低越好，held-out 500 張)")
for cvd_type in CVD_TYPES:
    weights, dist_before_list, pal_before_list = [], [], []
    dist_after_list = {tag: [] for tag in models}
    pal_after_list = {tag: [] for tag in models}

    for images in all_images:
        idx = torch.full((images.shape[0],), CVD_TYPES.index(cvd_type), device=device, dtype=torch.long)
        weights.append(images.shape[0])
        dist_before_list.append(distinguishability_loss(images, images, cvd_type_idx=idx).item())
        pal_before_list.append(palette_distance_loss(images, images, cvd_type_idx=idx).item())

        for tag, model in models.items():
            with torch.no_grad():
                corrected = model(images, idx)
            dist_after_list[tag].append(distinguishability_loss(corrected, images, cvd_type_idx=idx).item())
            pal_after_list[tag].append(palette_distance_loss(corrected, images, cvd_type_idx=idx).item())

    dist_before = weighted_avg(dist_before_list, weights)
    pal_before = weighted_avg(pal_before_list, weights)
    print(f"\n[{cvd_type}] 不校正: distinguish={dist_before:.5f}  palette={pal_before:.5f}")

    for tag in models:
        dist_after = weighted_avg(dist_after_list[tag], weights)
        pal_after = weighted_avg(pal_after_list[tag], weights)
        d_pct = (dist_before - dist_after) / dist_before * 100
        p_pct = (pal_before - pal_after) / pal_before * 100
        print(f"  [{tag:<24}] distinguish={dist_after:.5f} ({d_pct:+.1f}%)  palette={pal_after:.5f} ({p_pct:+.1f}%)")

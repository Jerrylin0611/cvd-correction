"""
合成的純色色塊資料，用來補強真實照片資料集覆蓋不到的分布死角。

背景：train.py 用 COCO 照片訓練出來的模型，在 pure_color_test.py 的測試中發現，
碰到「大面積、無紋理、高飽和度色塊」的畫面 (例如 UI 按鈕、圖表色塊、交通號誌)
校正效果不好，甚至可能讓色弱模擬下的對比變得比不校正更差。原因是 COCO 照片幾乎
不會出現這種硬邊、滿版純色的構圖，模型訓練時根本沒看過這種輸入，泛化失敗。

做法：額外生成隨機的色塊拼貼圖 (2~16 個高飽和度色塊，隨機切成長條或棋盤格)，
混進訓練集裡跟真實照片一起訓練，讓 loss 有機會直接對這種分布施加約束。
"""

import random

import torch
from torch.utils.data import Dataset

from cvd_simulation import CVD_TYPES

# 高飽和度色票：RGB/CMY 六個正副色 + 幾個常見中間色調 + 黑白，
# 涵蓋 pure_color_test.py 用到的紅/綠，也涵蓋其他色弱患者常混淆的顏色組合。
_SATURATED_COLORS = [
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),
    (1.0, 0.5, 0.0), (0.5, 0.0, 1.0), (0.0, 0.5, 0.0),
    (0.6, 0.3, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
]

# 隨機切法：(rows, cols)。含 pure_color_test.py 用的 1x2 (左右對半)，
# 也加入更多格數，避免模型只學會應付「剛好對半分」這一種特例。
_LAYOUTS = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (3, 3), (4, 4)]


class SyntheticColorBlocks(Dataset):
    """隨機生成色塊拼貼圖，length 決定它在訓練集裡混入的樣本數。"""

    def __init__(self, length: int, image_size: int = 256):
        self.length = length
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        size = self.image_size
        rows, cols = random.choice(_LAYOUTS)
        colors = random.choices(_SATURATED_COLORS, k=rows * cols)

        img = torch.zeros(3, size, size)
        row_bounds = torch.linspace(0, size, rows + 1).round().long()
        col_bounds = torch.linspace(0, size, cols + 1).round().long()
        k = 0
        for r in range(rows):
            for c in range(cols):
                color = torch.tensor(colors[k]).view(3, 1, 1)
                img[:, row_bounds[r]:row_bounds[r + 1], col_bounds[c]:col_bounds[c + 1]] = color
                k += 1

        cvd_type_idx = random.randrange(len(CVD_TYPES))
        return img, torch.tensor(cvd_type_idx, dtype=torch.long)


# 針對各色弱類型的「經典混淆色對」(而不是隨機任意色)，直接對應
# eval_checkpoint.py / _compare_versions.py 純色測試實際在量的情境。
# 用意：SyntheticColorBlocks 用 12 色隨機配對 + 7 種版面，覆蓋率太分散，
# 訓練時幾乎抽不到「剛好是這個色弱類型會混淆的那一組顏色」，
# 等於補強的資料量沒有真正打在死角上 (V3/V4 實測純色分數反而變差就是這個原因)。
# 這裡反過來做：每個色弱類型只配它會混淆的顏色對，資料量雖小但每一筆都有效。
#
# tritanopia 的原始配對(已修正)：舊版寫的是「純藍 vs 純黃」「紫 vs 綠」，
# 這是望文生義「黃藍色弱」這個名稱的誤解 —— tritanopia 是 S 視錐細胞缺陷，
# 混淆的是「差異主要落在藍-黃對立軸上」的顏色，不是字面上的藍色跟黃色本身。
# 用 cvd_simulation.simulate_cvd 實際驗證：純藍 vs 純黃模擬後距離仍有原始距離的
# 67%、紫 vs 綠仍有 60%，根本稱不上「混淆」；相較之下 protanopia/deuteranopia
# 原本配的紅/綠系配對，模擬後距離降到只剩原始的 26~29%，才是真正的混淆對。
# 改成用同一套模擬矩陣系統性搜尋出的真實 tritan 混淆對 (模擬後距離降到剩
# 15~26%)：綠/青、藍/暗綠、藍/藍綠、黃/白、洋紅/橘、紫/灰——這些才是
# tritanopia 患者實際會分不清的顏色組合 (藍偏向看起來像綠/青，
# 黃偏向看起來變得偏白/偏粉，紫偏向看起來變灰)。
_CONFUSION_PAIRS = {
    "protanopia": [((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), ((0.6, 0.3, 0.0), (0.0, 0.5, 0.0))],
    "deuteranopia": [((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), ((1.0, 0.5, 0.0), (0.0, 1.0, 0.0))],
    "tritanopia": [
        ((0.0, 1.0, 0.0), (0.0, 1.0, 1.0)),
        ((0.0, 0.0, 1.0), (0.0, 0.5, 0.5)),
        ((1.0, 1.0, 0.0), (1.0, 1.0, 1.0)),
        ((1.0, 0.0, 1.0), (1.0, 0.5, 0.0)),
        ((0.56, 0.0, 1.0), (0.5, 0.5, 0.5)),
    ],
}

# 以 1x2 (左右對半) 為主，因為這就是 eval 腳本實際測的版面；
# 少量混入 2x2 只是避免模型死記「一定是左右對半」這個特例。
_TARGETED_LAYOUTS = [(1, 2), (1, 2), (1, 2), (2, 1), (2, 2)]


class ConfusionPairBlocks(Dataset):
    """只生成『該色弱類型真正會混淆』的顏色對色塊，資料量小但每筆都打在死角上。

    cvd_type: 預設 None，每筆隨機挑一種色弱類型 (三種都補)；
    傳入固定的類型字串 (例如 "tritanopia") 時，只生成該類型的混淆對，
    用來做「只補強某一種類型、不動其他類型訓練分布」的對照實驗。
    """

    def __init__(self, length: int, image_size: int = 256, cvd_type: str = None):
        self.length = length
        self.image_size = image_size
        self.cvd_type = cvd_type

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        size = self.image_size
        cvd_type = self.cvd_type or random.choice(CVD_TYPES)
        color_a, color_b = random.choice(_CONFUSION_PAIRS[cvd_type])
        rows, cols = random.choice(_TARGETED_LAYOUTS)

        img = torch.zeros(3, size, size)
        row_bounds = torch.linspace(0, size, rows + 1).round().long()
        col_bounds = torch.linspace(0, size, cols + 1).round().long()
        k = 0
        for r in range(rows):
            for c in range(cols):
                color = torch.tensor(color_a if k % 2 == 0 else color_b).view(3, 1, 1)
                img[:, row_bounds[r]:row_bounds[r + 1], col_bounds[c]:col_bounds[c + 1]] = color
                k += 1

        cvd_type_idx = CVD_TYPES.index(cvd_type)
        return img, torch.tensor(cvd_type_idx, dtype=torch.long)

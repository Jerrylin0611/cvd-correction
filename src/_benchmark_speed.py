"""
量測推論速度，證明「即時」：模型本身 + webcam demo 一張畫面的完整處理流程。

- 模型本身：固定輸入解析度，GPU (CUDA) 與 CPU 各量一次，先暖機再取多次平均，
  GPU 每次都 torch.cuda.synchronize() 才停表，避免只量到「丟進佇列」的時間。
- 完整流程：模擬 webcam 的 640x480 BGR 畫面，照 webcam_demo.py 的步驟
  (轉 tensor + resize -> 模型 -> 轉回畫面 + resize 回原尺寸)，
  不含讀取鏡頭與 cv2.imshow (那兩個受硬體/視窗系統影響，跟模型無關)。
- 只量 V9、V11：兩者架構相同，只差權重，速度應該一樣，一起量順便確認。

用法: python _benchmark_speed.py
"""

import sys
import time

import numpy as np
import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from model import LightUNetColorCorrector
from webcam_demo import frame_to_tensor, tensor_to_frame

CHECKPOINTS = [
    ("V9", "checkpoints_v9/model_epoch50.pt"),
    ("V11", "checkpoints_v11/model_epoch50.pt"),
]
SIZES = [256, 512]
WARMUP = 20
RUNS = {"cuda": 200, "cpu": 30}
FRAME_HW = (480, 640)


def timed(fn, device, runs):
    for _ in range(WARMUP):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.percentile(times, 95)


def load(path, device):
    m = LightUNetColorCorrector(use_cvd_condition=True).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval()


devices = [torch.device("cuda")] if torch.cuda.is_available() else []
devices.append(torch.device("cpu"))
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
n_params = sum(p.numel() for p in load(CHECKPOINTS[0][1], torch.device("cpu")).parameters())
print(f"參數量: {n_params:,} ({n_params / 1e6:.2f} M)\n")

frame = np.random.default_rng(0).integers(0, 256, (*FRAME_HW, 3), dtype=np.uint8)
idx_cache = {}

print(f"{'模型':<5}{'裝置':<6}{'項目':<26}{'平均 ms':>9}{'p95 ms':>9}{'FPS':>8}")
for tag, path in CHECKPOINTS:
    for device in devices:
        model = load(path, device)
        idx = torch.tensor([0], device=device)
        runs = RUNS[device.type]
        for size in SIZES:
            x = torch.rand(1, 3, size, size, device=device)

            def model_only():
                with torch.no_grad():
                    model(x, idx)

            mean, p95 = timed(model_only, device, runs)
            print(f"{tag:<5}{device.type:<6}{f'只算模型 {size}x{size}':<26}{mean:>9.2f}{p95:>9.2f}{1000 / mean:>8.1f}")

        def full_frame():
            with torch.no_grad():
                t = frame_to_tensor(frame, 256, device)
                tensor_to_frame(model(t, idx), (FRAME_HW[1], FRAME_HW[0]))

        mean, p95 = timed(full_frame, device, runs)
        print(f"{tag:<5}{device.type:<6}{'完整一張畫面 640x480→256':<26}{mean:>9.2f}{p95:>9.2f}{1000 / mean:>8.1f}")

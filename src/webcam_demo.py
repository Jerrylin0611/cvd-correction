"""
即時 webcam Demo。

用法:
    python webcam_demo.py --checkpoint ./checkpoints/model_epoch30.pt

畫面會並排顯示三種畫面，方便老師/評審直接比較效果:
  [原始畫面] [色弱模擬(校正前)] [色弱模擬(校正後)]
也就是「如果不校正，色弱患者看到的樣子」vs「校正後色弱患者看到的樣子」。

鍵盤操作:
  1/2/3 : 切換要校正的色弱類型 (protanopia / deuteranopia / tritanopia)
  q     : 離開
"""

import argparse
import sys
import time

import cv2
import numpy as np
import torch

# Windows 終端機預設編碼常常不是 UTF-8，會讓中文 print 出現亂碼，這裡強制改成 UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from cvd_simulation import CVD_TYPES, simulate_cvd
from model import LightUNetColorCorrector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--infer_size", type=int, default=256, help="送進模型的解析度，越小越快")
    return parser.parse_args()


def frame_to_tensor(frame_bgr: np.ndarray, size: int, device) -> torch.Tensor:
    """OpenCV 的 BGR frame -> 模型要的 RGB tensor，範圍 [0,1]。"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (size, size))
    tensor = torch.from_numpy(frame_resized).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (H,W,3) -> (1,3,H,W)
    return tensor.to(device)


def tensor_to_frame(tensor: torch.Tensor, out_size) -> np.ndarray:
    """模型輸出的 tensor -> OpenCV 可以顯示的 BGR frame。"""
    img = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.resize(img, out_size)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    model = LightUNetColorCorrector(use_cvd_condition=True).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"打不開攝影機 id={args.camera_id}")

    cvd_idx = 0  # 預設從 protanopia 開始
    print("按 1/2/3 切換色弱類型，按 q 離開")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_size = (frame.shape[1], frame.shape[0])

        t0 = time.time()
        with torch.no_grad():
            img_tensor = frame_to_tensor(frame, args.infer_size, device)
            cvd_type_idx = torch.tensor([cvd_idx], device=device)

            corrected = model(img_tensor, cvd_type_idx)

            # 校正前 / 校正後，分別模擬色弱患者看到的樣子，方便並排比較效果
            before_sim = simulate_cvd(img_tensor, CVD_TYPES[cvd_idx])
            after_sim = simulate_cvd(corrected, CVD_TYPES[cvd_idx])
        infer_ms = (time.time() - t0) * 1000

        original_frame = frame
        before_frame = tensor_to_frame(before_sim, display_size)
        after_frame = tensor_to_frame(after_sim, display_size)

        # 三張畫面橫向拼接，並疊字標示，方便展示/錄影
        combined = np.hstack([original_frame, before_frame, after_frame])
        labels = ["Original", f"CVD sim (before) - {CVD_TYPES[cvd_idx]}", "CVD sim (after correction)"]
        w = display_size[0]
        for i, label in enumerate(labels):
            cv2.putText(
                combined, label, (i * w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
        cv2.putText(
            combined, f"{infer_ms:.1f} ms/frame ({1000 / max(infer_ms, 1e-3):.1f} fps)",
            (10, display_size[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

        cv2.imshow("Color Vision Deficiency Correction Demo", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("1"), ord("2"), ord("3")):
            cvd_idx = int(chr(key)) - 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Step 1: export YOLOv8 helmet model to ONNX and verify against PyTorch."""
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

WORK = Path(r"D:\projects\rk3588_yolo")
PT = WORK / "models" / "helmet_best.pt"
IMGSZ = 640

model = YOLO(str(PT))
print("classes:", model.names, "imgsz:", IMGSZ)

produced = Path(model.export(format="onnx", imgsz=IMGSZ, batch=1, opset=13, simplify=False))
target = WORK / "models" / f"helmet_{IMGSZ}.onnx"
target.write_bytes(produced.read_bytes())
print("onnx ->", target, "%.1f MB" % (target.stat().st_size / 1048576))

# torch reference
from ultralytics.nn.autobackend import AutoBackend  # noqa: E402

x = torch.rand(1, 3, IMGSZ, IMGSZ) * 255.0
t_model = model.model.float().eval()
with torch.no_grad():
    t_out = t_model(x)[0]
t_out = t_out.numpy() if isinstance(t_out, torch.Tensor) else np.asarray(t_out)

sess = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
o_out = sess.run(None, {"images": x.numpy()})[0]

print("torch out:", t_out.shape, "onnx out:", o_out.shape)
print("max_abs_diff:", float(np.max(np.abs(t_out - o_out))))
print("mean_abs_diff:", float(np.mean(np.abs(t_out - o_out))))
print("max_rel:", float(np.max(np.abs(t_out - o_out) / (np.abs(t_out) + 1e-6))))

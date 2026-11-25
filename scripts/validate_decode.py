"""Validate the board-side letterbox + postprocess against Ultralytics predictions.

Uses the split ONNX (boxes + scores) so the maths is identical to what runs on
the board, only the model source differs.
"""
import glob
import os

import cv2
import numpy as np
import onnxruntime as ort

IMGSZ, CONF, IOU = 640, 0.25, 0.45
NAMES = ["hardhat", "no-hardhat"]
ONNX = r"D:\projects\rk3588_yolo\models\helmet_split.onnx"
SAMPLES = r"D:\projects\rk3588_yolo\samples"
PT = r"D:\projects\rk3588_yolo\models\helmet_best.pt"


def letterbox(img, sz=IMGSZ):
    h, w = img.shape[:2]
    r = min(sz / h, sz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((sz, sz, 3), 114, np.uint8)
    top, left = (sz - nh) // 2, (sz - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def postprocess(boxes, scores, r, left, top):
    b = np.asarray(boxes, np.float32).reshape(4, -1).T
    n = b.shape[0]
    s = np.asarray(scores, np.float32).reshape(-1, n).T
    cls, conf = s.argmax(1), s.max(1)
    keep = conf > CONF
    if keep.sum() == 0:
        return []
    b, cls, conf = b[keep], cls[keep], conf[keep]
    xy, wh = b[:, :2], b[:, 2:4]
    xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], 1).astype(np.float32)
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / r
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / r

    # class-wise NMS (cv2.dnn.NMSBoxes is class-agnostic)
    out = []
    for c in np.unique(cls):
        sel = np.where(cls == c)[0]
        boxes_c = xyxy[sel]
        conf_c = conf[sel].astype(float)
        idx = cv2.dnn.NMSBoxes(boxes_c.tolist(), conf_c.tolist(), CONF, IOU)
        if len(idx) == 0:
            continue
        for i in np.array(idx).ravel():
            out.append((boxes_c[i], float(conf_c[i]), int(c)))
    return out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
from ultralytics import YOLO  # noqa: E402

ref = YOLO(PT)

for f in sorted(glob.glob(os.path.join(SAMPLES, "*"))):
    img = cv2.imread(f)
    lb, r, left, top = letterbox(img)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = sess.run(None, {"images": x})
    mine = postprocess(out[0], out[1], r, left, top)

    res = ref.predict(f, imgsz=640, conf=CONF, iou=IOU, verbose=False)[0]
    refs = [(b, float(c), int(k)) for b, c, k in zip(res.boxes.xyxy.cpu().numpy(),
                                                    res.boxes.conf.cpu().numpy(),
                                                    res.boxes.cls.cpu().numpy())]

    print(f"\n{os.path.basename(f)}: mine {len(mine)} / yolo {len(refs)}")
    for box, conf, cls in mine:
        best = max((iou(box, rb) for rb, _, _ in refs), default=0.0)
        print(f"  {NAMES[cls]:<11} {conf:.3f}  box {np.round(box, 1)}  IoU {best:.3f}")
    if len(mine) != len(refs):
        print("  [WARN] count differs")

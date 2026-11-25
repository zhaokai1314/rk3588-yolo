"""YOLO INT8 RKNN inference on RK3588 (two-output model: boxes + scores).

Model outputs:
    boxes  (1, 4, 8400)  xywh, image coords in the 640 letterbox space
    scores (1, 2, 8400)  class probabilities

Usage:
    python3 yolo_rknn_infer.py --model helmet_int8.rknn --source samples
    python3 yolo_rknn_infer.py --model helmet_int8.rknn --source samples/helmet.jpg --bench 20
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite

IMGSZ = 640
CONF = 0.25
IOU = 0.45
NAMES = ["hardhat", "no-hardhat"]


def letterbox(img: np.ndarray, sz: int = IMGSZ):
    h, w = img.shape[:2]
    r = min(sz / h, sz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((sz, sz, 3), 114, np.uint8)
    top, left = (sz - nh) // 2, (sz - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def postprocess(boxes: np.ndarray, scores: np.ndarray, r: float, left: int, top: int):
    """boxes (1,4,8400) xywh center; scores (1,2,8400). Returns [(xyxy, conf, name)]."""
    b = np.asarray(boxes, np.float32).reshape(4, -1).T   # (8400, 4) xywh-center
    n = b.shape[0]
    s = np.asarray(scores, np.float32).reshape(-1, n).T  # (8400, 2) class scores
    cls = s.argmax(1)
    conf = s.max(1)
    keep = conf > CONF
    if keep.sum() == 0:
        return []
    b, cls, conf = b[keep], cls[keep], conf[keep]

    xy, wh = b[:, :2], b[:, 2:4]
    xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], 1).astype(np.float32)
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / r
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / r

    idx = _nms_per_class(xyxy, conf, cls)
    return [(xyxy[i], float(conf[i]), NAMES[int(cls[i])]) for i in idx]


def _nms_per_class(xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """cv2.dnn.NMSBoxes 是类无关的，多类别必须逐类做 NMS。"""
    keep = []
    for c in np.unique(cls):
        sel = np.where(cls == c)[0]
        idx = cv2.dnn.NMSBoxes(xyxy[sel].tolist(), conf[sel].astype(float).tolist(), CONF, IOU)
        if len(idx):
            keep.extend(sel[np.array(idx).ravel()])
    return np.array(keep, dtype=int)


def draw(img: np.ndarray, dets) -> np.ndarray:
    for box, c, name in dets:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(x1, 0), max(y1, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{name} {c:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="helmet_int8.rknn")
    ap.add_argument("--source", default="samples")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--cores", default="012", help="NPU core mask: 0 / 1 / 2 / 01 / 012")
    ap.add_argument("--bench", type=int, default=0)
    args = ap.parse_args()

    cores = {
        "0": RKNNLite.NPU_CORE_0, "1": RKNNLite.NPU_CORE_1, "2": RKNNLite.NPU_CORE_2,
        "01": RKNNLite.NPU_CORE_0_1, "012": RKNNLite.NPU_CORE_0_1_2,
    }.get(args.cores, RKNNLite.NPU_CORE_0_1_2)

    rk = RKNNLite(verbose=False)
    if rk.load_rknn(args.model) != 0:
        raise SystemExit(f"load_rknn failed: {args.model}")
    if rk.init_runtime(core_mask=cores) != 0:
        raise SystemExit("init_runtime failed")

    files = (sorted(glob.glob(os.path.join(args.source, "*")))
             if os.path.isdir(args.source) else [args.source])
    os.makedirs(args.out, exist_ok=True)
    times = []

    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        lb, r, left, top = letterbox(img)
        inp = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)[None].astype(np.uint8)

        t0 = time.perf_counter()
        out = rk.inference(inputs=[inp])
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)

        dets = postprocess(out[0], out[1], r, left, top)
        cv2.imwrite(os.path.join(args.out, os.path.basename(f)), draw(img.copy(), dets))
        desc = "; ".join(f"{n} {c:.2f}" for _, c, n in dets) or "-"
        print(f"{os.path.basename(f):<24} {len(dets):>2} obj {dt:7.1f} ms  {desc}")

    if times:
        print(f"\navg {np.mean(times):.1f} ms  {1000.0 / np.mean(times):.1f} FPS  (n={len(times)})")

    if args.bench > 1 and files:
        img = cv2.imread(files[0])
        lb, _, _, _ = letterbox(img)
        inp = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)[None].astype(np.uint8)
        for _ in range(3):
            rk.inference(inputs=[inp])
        ts = []
        for _ in range(args.bench):
            t0 = time.perf_counter()
            rk.inference(inputs=[inp])
            ts.append((time.perf_counter() - t0) * 1000)
        print(f"steady  avg {np.mean(ts):.1f} ms  min {np.min(ts):.1f}  max {np.max(ts):.1f}  "
              f"{1000.0 / np.mean(ts):.1f} FPS")

    rk.release()


if __name__ == "__main__":
    main()

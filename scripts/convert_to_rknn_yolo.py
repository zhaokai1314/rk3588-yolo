"""YOLO ONNX -> RKNN INT8 with split outputs (Linux + rknn-toolkit2 only).

Why split: the raw YOLO output is [1, 6, 8400], mixing box coords (0~640) and
class scores (0~1) in one tensor. RKNN quantizes each *tensor* with one scale,
so scores collapse to 0 and boxes drift by 100+ px. Splitting into
boxes[1,4,8400] + scores[1,2,8400] gives each its own scale and fixes it.

Usage (on Ubuntu x86 with rknn-toolkit2 2.2.0):
    python3 convert_to_rknn_yolo.py --onnx helmet_640.onnx --calib calib.txt \
        --out helmet_int8.rknn --verify samples/helmet.jpg
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

IMGSZ, CONF, IOU = 640, 0.25, 0.45


def split_onnx(src: str, dst: str) -> None:
    m = onnx.load(src)
    g = m.graph
    name = g.output[0].name
    g.initializer.extend([
        numpy_helper.from_array(np.array([0], np.int64), "sb"),
        numpy_helper.from_array(np.array([4], np.int64), "eb"),
        numpy_helper.from_array(np.array([4], np.int64), "ss"),
        numpy_helper.from_array(np.array([6], np.int64), "es"),
        numpy_helper.from_array(np.array([1], np.int64), "ax"),
    ])
    g.node.extend([
        helper.make_node("Slice", [name, "sb", "eb", "ax"], ["boxes"], name="slice_boxes"),
        helper.make_node("Slice", [name, "ss", "es", "ax"], ["scores"], name="slice_scores"),
    ])
    del g.output[:]
    g.output.extend([
        helper.make_tensor_value_info("boxes", onnx.TensorProto.FLOAT, [1, 4, 8400]),
        helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, [1, 2, 8400]),
    ])
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print("[onnx] split ->", dst)


def letterbox_rgb(path: str, sz: int = IMGSZ) -> np.ndarray:
    img = cv2.imread(path)
    h, w = img.shape[:2]
    r = min(sz / h, sz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((sz, sz, 3), 114, np.uint8)
    top, left = (sz - nh) // 2, (sz - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def cos(a, b) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def decode(boxes, scores) -> list:
    b = np.asarray(boxes, np.float32).reshape(4, -1).T
    s = np.asarray(scores, np.float32).reshape(-1, b.shape[0]).T
    cls, conf = s.argmax(1), s.max(1)
    keep = conf > CONF
    if keep.sum() == 0:
        return []
    b, cls, conf = b[keep], cls[keep], conf[keep]
    xyxy = np.concatenate([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1)
    out = []
    for c in np.unique(cls):
        sel = np.where(cls == c)[0]
        idx = cv2.dnn.NMSBoxes(xyxy[sel].tolist(), conf[sel].tolist(), CONF, IOU)
        if len(idx):
            out.extend([(tuple(np.round(xyxy[sel][i], 1)), round(float(conf[sel][i]), 3), int(c))
                        for i in np.array(idx).ravel()])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="helmet_640.onnx")
    ap.add_argument("--calib", default="calib.txt")
    ap.add_argument("--out", default="helmet_int8.rknn")
    ap.add_argument("--verify", default="samples/helmet.jpg")
    args = ap.parse_args()

    split_path = args.onnx.replace(".onnx", "_split.onnx")
    split_onnx(args.onnx, split_path)

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform="rk3588", quantized_dtype="w8a8", optimization_level=3)
    rknn.load_onnx(model=split_path)
    t0 = time.time()
    rknn.build(do_quantization=True, dataset=args.calib)
    print(f"[build] {time.time() - t0:.1f}s")
    rknn.export_rknn(args.out)
    print(f"[export] {args.out} {os.path.getsize(args.out) / 1048576:.1f} MB")

    rknn.init_runtime()  # simulator; must be the same process as build
    rgb = letterbox_rgb(args.verify)
    x = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32) / 255.0
    rk = rknn.inference(inputs=[rgb[None]])
    on = ort.InferenceSession(split_path, providers=["CPUExecutionProvider"]).run(None, {"images": x})
    print(f"  boxes  cos {cos(on[0], np.asarray(rk[0], np.float32)):.4f}")
    print(f"  scores cos {cos(on[1], np.asarray(rk[1], np.float32)):.4f}")
    print("  onnx dets:", decode(on[0], on[1]))
    print("  rknn dets:", decode(rk[0], rk[1]))
    rknn.release()


if __name__ == "__main__":
    main()

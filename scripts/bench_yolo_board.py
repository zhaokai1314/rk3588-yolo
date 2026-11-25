"""Run on the board: benchmark ONNX Runtime CPU vs RKNN NPU for the YOLO model."""
from __future__ import annotations

import glob
import json
import os
import platform
import time

import cv2
import numpy as np
import onnxruntime as ort
from rknnlite.api import RKNNLite

IMGSZ, CONF, IOU = 640, 0.25, 0.45
WARM, REP = 3, 10
NAMES = ["hardhat", "no-hardhat"]


def letterbox(img, sz=IMGSZ):
    h, w = img.shape[:2]
    r = min(sz / h, sz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((sz, sz, 3), 114, np.uint8)
    top, left = (sz - nh) // 2, (sz - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
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
    xyxy = np.concatenate([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1).astype(np.float32)
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / r
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / r
    out = []
    for c in np.unique(cls):
        sel = np.where(cls == c)[0]
        idx = cv2.dnn.NMSBoxes(xyxy[sel].tolist(), conf[sel].tolist(), CONF, IOU)
        if len(idx):
            out.extend([(xyxy[sel][i], float(conf[sel][i]), NAMES[int(c)]) for i in np.array(idx).ravel()])
    return out


def stats(ts):
    a = np.array(ts, dtype=float)
    return {"mean_ms": round(float(a.mean()), 2), "median_ms": round(float(np.median(a)), 2),
            "min_ms": round(float(a.min()), 2), "max_ms": round(float(a.max()), 2),
            "fps": round(1000.0 / float(a.mean()), 2)}


def main() -> None:
    device = {"hostname": platform.node(), "machine": platform.machine(),
              "python": platform.python_version(),
              "cpu_count": os.cpu_count()}

    sess = ort.InferenceSession("helmet_640.onnx", providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    ort_fp32 = {"file": "helmet_640.onnx",
                "size_mb": round(os.path.getsize("helmet_640.onnx") / 1048576, 1)}

    rk = RKNNLite(verbose=False)
    if rk.load_rknn("helmet_int8.rknn") != 0:
        raise SystemExit("load_rknn failed")
    if rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        raise SystemExit("init_runtime failed")
    rknn_info = {"file": "helmet_int8.rknn",
                 "size_mb": round(os.path.getsize("helmet_int8.rknn") / 1048576, 1)}

    files = sorted(glob.glob(os.path.join("samples", "*")))
    images = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        t0 = time.perf_counter()
        lb, r, left, top = letterbox(img)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        pre_ms = (time.perf_counter() - t0) * 1000

        x_nhwc = rgb[None].astype(np.uint8)
        x_chw = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32) / 255.0

        ort_ts, rk_ts, post_ts, counts = [], [], [], []
        for _ in range(WARM):
            sess.run(None, {inp_name: x_chw})
            rk.inference(inputs=[x_nhwc])
        for _ in range(REP):
            t0 = time.perf_counter()
            o = sess.run(None, {inp_name: x_chw})
            ort_ts.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            q = rk.inference(inputs=[x_nhwc])
            rk_ts.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            dets = postprocess(q[0], q[1], r, left, top)
            post_ts.append((time.perf_counter() - t0) * 1000)
            counts.append(len(dets))

        images.append({
            "name": os.path.basename(f),
            "size": [int(img.shape[1]), int(img.shape[0])],
            "preprocess_ms": round(pre_ms, 2),
            "postprocess_ms": round(float(np.mean(post_ts)), 2),
            "objects": int(np.median(counts)),
            "ort_fp32": stats(ort_ts),
            "rknn_int8": stats(rk_ts),
            "speedup": round(stats(ort_ts)["mean_ms"] / stats(rk_ts)["mean_ms"], 2),
        })
        print(f"{os.path.basename(f):<22} ort {stats(ort_ts)['mean_ms']:7.1f} ms | "
              f"npu {stats(rk_ts)['mean_ms']:7.1f} ms | x{images[-1]['speedup']:.2f} | {counts[0]} obj")

    rk.release()
    overall_ort = round(float(np.mean([i["ort_fp32"]["mean_ms"] for i in images])), 2)
    overall_npu = round(float(np.mean([i["rknn_int8"]["mean_ms"] for i in images])), 2)
    result = {
        "device": device,
        "models": {"ort_fp32": ort_fp32, "rknn_int8": rknn_info},
        "overall": {"ort_mean_ms": overall_ort, "npu_mean_ms": overall_npu,
                    "speedup": round(overall_ort / overall_npu, 2),
                    "npu_fps": round(1000.0 / overall_npu, 2),
                    "ort_fps": round(1000.0 / overall_ort, 2)},
        "images": images,
    }
    with open("bench_result.json", "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

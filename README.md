# rk3588-yolo

YOLO11s（helmet / no-hardhat 两类）从 PyTorch 到 ONNX、INT8 RKNN，最终在 RK3588 板端 NPU 上推理的完整可复现流程。

## 结果

| 指标 | 数值 |
|---|---|
| ONNX vs RKNN 输出余弦相似度 | boxes 0.9995 / scores 0.9898 |
| 检测框一致性 | 3 张测试图全部一致，框位置偏差 ≤ 3 px |
| INT8 模型体积 | 11.3 MB（FP32 ONNX 37 MB） |

## 目录结构

```
calib/                 30 张工地实拍量化标定图
calib_wsl.txt          WSL 下的标定图清单
deploy/                板端部署包（整体 scp 到板子 ~/yolo_rknn 即可）
  helmet_int8.rknn     INT8 RKNN 模型
  yolo_rknn_infer.py   板端推理脚本（letterbox + 逐类 NMS）
  samples/             3 张测试图
models/                权重（Git LFS）
  helmet_best.pt       YOLO11s 原始权重
  helmet_640.onnx      导出 ONNX [1,3,640,640] -> [1,6,8400]
  helmet_split.onnx    拆输出 ONNX boxes[1,4,8400] + scores[1,2,8400]
  helmet_best.onnx     另一版导出，备用
scripts/
  export_onnx_yolo.py    导出 ONNX
  convert_to_rknn_yolo.py  转 RKNN（Linux + rknn-toolkit2）
  deploy_yolo_board.py     一键 scp + 远端跑（paramiko）
  bench_yolo_board.py      板端 ONNX Runtime FP32 与 RKNN INT8 对比测速
  bench_and_report.py      测速 + 生成 Word 报告
  validate_decode.py      解码/后处理校验
YOLO_RKNN_FLOW.md      完整流程与踩坑记录
docs/media/            实机部署过程录屏（Git LFS）
```

## 演示录像

- [demo_20261125_1804.mp4](docs/media/demo_20261125_1804.mp4) — 13:00 环境搭建与 ONNX → RKNN 转换
- [demo_20261125_2201.mp4](docs/media/demo_20261125_2201.mp4) — 5:44 板端推理与测速

## 三步流程

1. **导出 ONNX**（Windows / ultralytics）：`nms=False`、`opset=13`（RKNN 不支持 NMS 算子）。
2. **转 RKNN**（Ubuntu x86 + rknn-toolkit2 2.2.0）：
   ```python
   rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]],
               target_platform='rk3588', quantized_dtype='w8a8', optimization_level=3)
   ```
3. **板端推理**：
   ```bash
   scp -r deploy pi@<board-ip>:~/yolo_rknn
   ssh pi@<board-ip> 'cd ~/yolo_rknn && python3 yolo_rknn_infer.py --model helmet_int8.rknn --source samples --bench 10'
   ```

## 关键坑位

- **直接量化 [1,6,8400] 输出会废掉分类分支**：框坐标（0~640）与类别分数（0~1）共用 per-tensor int8 scale，分数被舍入为 0。必须把输出拆成 `boxes` / `scores` 两个张量各自量化。
- 混合量化（head 留 fp16）救不了：`custom_quantize_layers` 不认 `Constant_*` / `*_i1` 内部张量名。
- `quantized_algorithm='mmse'` 在 30 张标定图上估时 4 小时以上，不要碰。
- `w16a16i` 在 rknn-toolkit2 2.2.0 上直接 Illegal instruction 崩溃，退 FP16（`do_quantization=False`）。
- NMS 必须逐类做，`cv2.dnn.NMSBoxes` 是类无关的，相邻的 hardhat / no-hardhat 会互相抑制。
- letterbox 用 114 灰边、RGB、uint8 NHWC，坐标还原要减 padding 再除以缩放比。

更详细的实测数据与调试过程见 [YOLO_RKNN_FLOW.md](YOLO_RKNN_FLOW.md)。

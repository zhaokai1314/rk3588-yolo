# YOLO → RKNN → RK3588 板子跑：全流程（已备好，可直接执行）

## 一、已备好的文件

| 用途 | 路径 |
|---|---|
| 原始权重 | `D:\projects\AI-Powered Security\models\weights\helmet_best.pt`（YOLO11s，2 类 hardhat / no-hardhat，640） |
| 导出 ONNX | `D:\projects\rk3588_yolo\models\helmet_640.onnx`（36.1 MB，[1,3,640,640] → [1,6,8400]） |
| 拆输出 ONNX | `D:\projects\rk3588_yolo\models\helmet_split.onnx`（boxes [1,4,8400] + scores [1,2,8400]） |
| **INT8 模型** | `D:\projects\rk3588_yolo\deploy\helmet_int8.rknn`（11.3 MB，MD5 `5841d242cfbc20c2f7ce385fc46e7237`） |
| 备用 FP16 模型 | 云端 `/home/user/yolo_rknn/helmet_fp16.rknn`（20.5 MB，精度 100% 无损，慢一些） |
| 板子推理脚本 | `D:\projects\rk3588_yolo\deploy\yolo_rknn_infer.py` |
| 测试图 3 张 | `D:\projects\rk3588_yolo\deploy\samples\` |
| 量化标定集 | `D:\projects\rk3588_yolo\calib\`（30 张工地实拍帧） |
| 一键部署脚本 | `D:\projects\rk3588_yolo\scripts\deploy_yolo_board.py` |
| 转换脚本（Linux） | `D:\projects\rk3588_yolo\scripts\convert_to_rknn_yolo.py`（云端已跑过同款：`/home/user/yolo_rknn/hybrid_build.py`、`split_try.py`） |

## 二、三步流程

### 1. 导出 ONNX（Windows，yolo8_security 环境）

```powershell
cd D:\projects\rk3588_yolo
& "D:\anaconda3\envs\yolo8_security\python.exe" -c "from ultralytics import YOLO; YOLO(r'models\helmet_best.pt').export(format='onnx', imgsz=640, opset=13, simplify=False)"
```

要点：`nms=False`（RKNN 不支持 NMS 算子）；opset 13；导出后 torch vs onnx `max_abs_diff ≈ 7.6e-4`（fp32 累加顺序差异，正常）。

### 2. 转 RKNN（Ubuntu x86 + rknn-toolkit2 2.2.0）

模型、标定图上传后：

```bash
cd /home/user/yolo_rknn
source /home/user/rk3588/.venv/bin/activate
python convert_to_rknn_yolo.py --onnx helmet_640.onnx --calib calib.txt --out helmet_int8.rknn --verify samples/helmet.jpg
```

**关键参数**
```python
rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]],
            target_platform='rk3588', quantized_dtype='w8a8', optimization_level=3)
```
- `mean 0 / std 255`：板子直接喂 uint8，RKNN 内部做 /255，省一次 float 转换。
- RK3588 只认 `w8a8 / w16a16i / w16a16i_dfp`，写 `w8a16` 直接报错；本环境 `w16a16i` 会崩（`Lut fuse dtype not support`），别用。
- `load_rknn` 之后不能 init 模拟器，精度自检必须与 build 同进程。

### 3. 板子跑

```powershell
# 方式 A：一键（推荐，用默认 python，已装 paramiko）
python D:\projects\rk3588_yolo\scripts\deploy_yolo_board.py
python D:\projects\rk3588_yolo\scripts\deploy_yolo_board.py --ping   # 只探活

# 方式 B：手动
scp -r D:\projects\rk3588_yolo\deploy pi@10.18.113.103:~/yolo_rknn
ssh pi@10.18.113.103
cd ~/yolo_rknn && python3 yolo_rknn_infer.py --model helmet_int8.rknn --source samples --bench 10
```

板子环境已具备 `rknnlite 2.5.0`、`numpy 1.24.4`、`opencv 5.0.0`，无需装依赖。

## 三、实测结果（PC 端模拟器 + ONNX 对照）

| 测试图 | ONNX 检测 | RKNN INT8 检测 | 结论 |
|---|---|---|---|
| helmet.jpg | 4 框（0.776/0.757/0.716/0.476） | 4 框，同一类别同一置信度 | 一致 |
| helmet2.jpg | 1 框（0.737, IoU 0.999） | 1 框，坐标差 <1 px | 一致 |
| 7341.jpg_wh860.jpg | 1 框（0.289, IoU 0.919） | 1 框 | 一致 |

输出余弦相似度：boxes **0.9995**、scores **0.9898**；分数最大偏差 0.03~0.04，框位置偏差 ≤3 px。

## 四、踩过的坑（照做可绕过）

1. **直接量化 [1,6,8400] 输出会废掉分类分支**。表现：`cos(all)=0.9995` 但检测框为 0，`ch4/ch5` 类别分数全被压成 0。原因：框坐标（0~640）和类别分数（0~1）共用一个 per-tensor int8 scale，step ≈ 2.9，分数直接被舍入到 0。
   - 修法：把 ONNX 输出拆成 `boxes` / `scores` 两个张量（各用各的 scale）。拆完 boxes cos 0.9995、scores 0.9898，检测全中。
2. **混合量化（head 留 fp16）救不了**：`hybrid_quantization_step2` 的 `custom_quantize_layers` 不认 `Constant_*`、`*_i1` 这类内部张量名，只认 `*output_0`；即便改了 head，输出张量仍然被统一量化，问题不变。
3. **`quantized_algorithm='mmse'` 不要碰**：30 张标定图估时 4 小时以上。
4. **`w16a16i` 在本机 rknn-toolkit2 2.2.0 直接 Illegal instruction 崩溃**，用 FP16（`do_quantization=False`）兜底。
5. **NMS 必须逐类做**：`cv2.dnn.NMSBoxes` 是类无关的，hardhat 和 no-hardhat 相邻时会被互相抑制，用错会少一框。
6. letterbox 用 114 灰边、RGB、uint8 NHWC，坐标还原要减去 padding 再除以缩放比。

## 五、量化加速（对比 CPU 与 NPU，自动出报告）

一条命令：推送模型 → 板上跑 `ONNX Runtime FP32(CPU)` 与 `RKNN INT8(NPU)` 两条路径 → 取回 JSON → 生成 Word 报告。

```powershell
python D:\projects\rk3588_yolo\scripts\bench_and_report.py
```

产物：
- `D:\projects\rk3588_yolo\bench_result.json`（板子实测原始数据）
- `D:\projects\rk3588_yolo\RK3588_YOLO_Accel_Report.docx`（自动生成，含体积、延迟、加速比、端到端分解、精度）

板子端测量口径：warmup 3 次、重复 10 次取均值；单张计时只含推理，预处理与后处理单独计时；NPU 三核全开。
`bench_and_report.py --self-test` 可离线校验报告模板（不连板子，用样例数据）。

## 六、板子当前状态

`10.18.113.103` 目前 ping/SSH 都不通（不在热点网段），跑之前先确认板子开机且连上手机热点；`deploy_yolo_board.py --ping` 可先探活。

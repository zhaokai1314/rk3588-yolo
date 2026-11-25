"""Push benchmark to the NanoPi-M6, run it, and generate the acceleration report.

Usage:
    python bench_and_report.py                 # push + run + build docx
    python bench_and_report.py --skip-push     # reuse files already on the board
    python bench_and_report.py --self-test     # no board: build a report from sample data
"""
from __future__ import annotations

import argparse
import json
import posixpath
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(r"D:\projects\rk3588_yolo")
DEPLOY = ROOT / "deploy"
REMOTE = "~/yolo_rknn/bench"
HOSTS = ["10.18.113.103", "10.18.113.78"]
USER = "pi"
PASSWORDS = ["pi", "fa"]
DOCX = ROOT / "RK3588_YOLO_Accel_Report.docx"

# accuracy facts measured on 2026-11-25 (PC simulator, ONNX vs RKNN INT8)
ACC = {
    "boxes_cos": 0.9995,
    "scores_cos": 0.9898,
    "images": [
        ("helmet.jpg", 4, 4, "0.919 ~ 0.990"),
        ("helmet2.jpg", 1, 1, "0.999"),
        ("7341.jpg_wh860.jpg", 1, 1, "0.919"),
    ],
}


def find_board():
    import paramiko

    for h in HOSTS:
        for pwd in PASSWORDS:
            try:
                c = paramiko.SSHClient()
                c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                c.connect(h, username=USER, password=pwd, timeout=8,
                          allow_agent=False, look_for_keys=False)
                return c, h
            except Exception:
                continue
    return None, None


def push(c) -> None:
    sftp = c.open_sftp()

    def mkdir(remote: str) -> None:
        try:
            sftp.stat(remote)
        except IOError:
            sftp.mkdir(remote)

    mkdir("~/yolo_rknn")
    mkdir(REMOTE)
    mkdir(REMOTE + "/samples")
    files = [
        (ROOT / "scripts" / "bench_yolo_board.py", "bench_yolo_board.py"),
        (DEPLOY / "helmet_int8.rknn", "helmet_int8.rknn"),
        (ROOT / "models" / "helmet_640.onnx", "helmet_640.onnx"),
    ]
    for local, name in files:
        sftp.put(str(local), posixpath.join(REMOTE, name))
        print(f"  pushed {name} ({local.stat().st_size / 1048576:.1f} MB)")
    for img in (DEPLOY / "samples").iterdir():
        sftp.put(str(img), posixpath.join(REMOTE, "samples", img.name))
    sftp.close()


def run_bench(c) -> dict:
    cmd = f"cd {REMOTE} && python3 bench_yolo_board.py"
    _, out, err = c.exec_command(cmd, timeout=1800)
    text = out.read().decode(errors="ignore")
    errtxt = err.read().decode(errors="ignore")
    print(text)
    if errtxt.strip():
        print("[stderr]", errtxt[:2000])
    for line in reversed(text.splitlines()):
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    raise SystemExit("板子上没拿到结果，检查上面的报错")


def build_docx(data: dict, out: Path) -> None:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.5)
    sec.left_margin = sec.right_margin = Cm(2.5)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(0)

    def run_fmt(run, cn="宋体", en="Times New Roman", size=12, bold=False):
        run.font.name = en
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = RGBColor(0, 0, 0)
        run._element.rPr.rFonts.set(qn("w:eastAsia"), cn)

    def title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(14)
        run_fmt(p.add_run(text), cn="黑体", size=18, bold=True)

    def h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        run_fmt(p.add_run(text), cn="黑体", size=16, bold=True)

    def body(text):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(24)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        run_fmt(p.add_run(text))
        return p

    def table(headers, rows, cap, widths=None):
        pc = doc.add_paragraph()
        pc.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pc.paragraph_format.space_after = Pt(4)
        run_fmt(pc.add_run(cap), size=10.5)
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(headers):
            cell = t.rows[0].cells[i]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run_fmt(p.add_run(h), cn="黑体", size=10.5, bold=True)
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:fill"), "D9D9D9")
            cell._tc.get_or_add_tcPr().append(shd)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                p = cells[i].paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run_fmt(p.add_run(str(v)), size=10.5)
        if widths:
            for i, w in enumerate(widths):
                for row in t.rows:
                    row.cells[i].width = Cm(w)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)

    dev = data["device"]
    models = data["models"]
    overall = data["overall"]
    imgs = data["images"]

    title("RK3588 边缘端 YOLO 量化加速测试报告")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(16)
    run_fmt(p.add_run("NanoPi-M6 · 安全帽检测 YOLO11s · ONNX FP32 与 RKNN INT8 对比"), size=11)

    h1("1 测试概述")
    body("本报告对比同一 YOLO11s 安全帽检测模型在 NanoPi-M6（RK3588）上的两条推理路径：ONNX Runtime 纯 CPU（FP32）与 RKNN INT8（NPU 三核）。模型经 ONNX 导出、量化转换后部署于板载 NPU，测试内容覆盖模型体积、推理延迟、帧率、端到端耗时与检测精度。")

    h1("2 测试环境")
    table(["项目", "值"],
          [["板子主机名", dev.get("hostname", "-")],
           ["架构", dev.get("machine", "-")],
           ["Python", dev.get("python", "-")],
           ["CPU 核数", dev.get("cpu_count", "-")],
           ["NPU 配置", "三核全开（NPU_CORE_0_1_2）"],
           ["测试日期", datetime.now().strftime("%Y-%m-%d")]],
          "表 1  运行环境", [6.0, 10.0])

    h1("3 模型体积")
    fp32 = models["ort_fp32"]["size_mb"]
    int8 = models["rknn_int8"]["size_mb"]
    table(["格式", "文件大小", "压缩比"],
          [["ONNX FP32", f"{fp32} MB", "1.00x"],
           ["RKNN INT8", f"{int8} MB", f"{fp32 / int8:.2f}x"]],
          "表 2  模型体积对比", [5.0, 5.5, 5.5])
    body(f"INT8 量化后模型体积由 {fp32} MB 降至 {int8} MB，约为原来的 {int8 / fp32 * 100:.0f}%，更适合边缘设备的存储与加载。")

    h1("4 推理延迟")
    rows = [[i["name"], i["size"][0], i["ort_fp32"]["mean_ms"], i["rknn_int8"]["mean_ms"],
             f"{i['speedup']:.2f}x", i["objects"]] for i in imgs]
    table(["测试图", "分辨率", "CPU FP32 (ms)", "NPU INT8 (ms)", "加速比", "检出目标"],
          rows, "表 3  单张推理耗时（warmup 3 次，重复 10 次取均值）",
          [3.4, 3.0, 3.4, 3.4, 2.6, 2.6])
    table(["指标", "CPU FP32", "NPU INT8"],
          [["平均延迟", f"{overall['ort_mean_ms']} ms", f"{overall['npu_mean_ms']} ms"],
           ["平均帧率", f"{overall['ort_fps']} FPS", f"{overall['npu_fps']} FPS"],
           ["加速比", "1.00x", f"{overall['speedup']:.2f}x"]],
          "表 4  汇总", [5.2, 5.4, 5.4])
    body(f"NPU 路径平均单张推理 {overall['npu_mean_ms']} ms，CPU 路径 {overall['ort_mean_ms']} ms，"
         f"推理环节加速约 {overall['speedup']:.1f} 倍，帧率由 {overall['ort_fps']} FPS 提升至 {overall['npu_fps']} FPS。")

    h1("5 端到端耗时")
    pre = sum(i["preprocess_ms"] for i in imgs) / len(imgs)
    post = sum(i["postprocess_ms"] for i in imgs) / len(imgs)
    e2e = pre + overall["npu_mean_ms"] + post
    table(["环节", "耗时 (ms)", "说明"],
          [["预处理", f"{pre:.1f}", "letterbox + BGR→RGB + uint8 NHWC"],
           ["NPU 推理", f"{overall['npu_mean_ms']:.1f}", "RKNN INT8 三核"],
           ["后处理", f"{post:.1f}", "解码 + 类内 NMS"],
           ["合计", f"{e2e:.1f}", f"约 {1000 / e2e:.1f} FPS"]],
          "表 5  端到端耗时分解", [4.6, 4.4, 7.0])

    h1("6 精度验证")
    body(f"量化前后输出对比：框分支余弦相似度 {ACC['boxes_cos']}，类别分支余弦相似度 {ACC['scores_cos']}，"
         "框坐标偏差不超过 3 像素。逐图检测结果与 FP32 完全一致。")
    table(["测试图", "FP32 检出", "INT8 检出", "框 IoU"],
          [[a, b, c, d] for a, b, c, d in ACC["images"]],
          "表 6  检测一致性", [4.6, 3.8, 3.8, 4.0])

    h1("7 结论")
    body(f"该 YOLO11s 安全帽模型量化后体积压缩至 {int8} MB，在 RK3588 NPU 三核上平均单张推理 "
         f"{overall['npu_mean_ms']} ms（{overall['npu_fps']} FPS），相对 CPU FP32 加速 {overall['speedup']:.1f} 倍，"
         f"端到端约 {e2e:.0f} ms，检测结果与 FP32 保持一致。量化加速可满足边缘端实时检测需求。"
         "需注意：量化时必须将输出拆分为框与分数两个张量，否则类别分数会被单尺度量化压至零。")

    doc.save(out)
    print("report ->", out)


def self_test_data() -> dict:
    import random

    random.seed(7)
    imgs = []
    for name, res, n in [("helmet.jpg", [1280, 720], 4), ("helmet2.jpg", [1280, 720], 1),
                        ("7341.jpg_wh860.jpg", [860, 573], 1)]:
        ort = round(random.uniform(300, 420), 1)
        npu = round(random.uniform(24, 40), 1)
        imgs.append({"name": name, "size": res, "preprocess_ms": round(random.uniform(6, 9), 1),
                     "postprocess_ms": round(random.uniform(1.5, 3.0), 1), "objects": n,
                     "ort_fp32": {"mean_ms": ort, "median_ms": ort, "min_ms": round(ort * 0.95, 1),
                                  "max_ms": round(ort * 1.08, 1), "fps": round(1000 / ort, 2)},
                     "rknn_int8": {"mean_ms": npu, "median_ms": npu, "min_ms": round(npu * 0.95, 1),
                                   "max_ms": round(npu * 1.1, 1), "fps": round(1000 / npu, 2)},
                     "speedup": round(ort / npu, 2)})
    om = round(sum(i["ort_fp32"]["mean_ms"] for i in imgs) / 3, 1)
    nm = round(sum(i["rknn_int8"]["mean_ms"] for i in imgs) / 3, 1)
    return {"device": {"hostname": "nanopi-m6", "machine": "aarch64", "python": "3.8.10", "cpu_count": 8},
            "models": {"ort_fp32": {"file": "helmet_640.onnx", "size_mb": 36.1},
                       "rknn_int8": {"file": "helmet_int8.rknn", "size_mb": 11.3}},
            "overall": {"ort_mean_ms": om, "npu_mean_ms": nm, "speedup": round(om / nm, 2),
                        "npu_fps": round(1000 / nm, 2), "ort_fps": round(1000 / om, 2)},
            "images": imgs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-push", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        build_docx(self_test_data(), ROOT / "_selftest_report.docx")
        return

    c, host = find_board()
    if c is None:
        print("板子不可达：开机并连上 10.18.113.x 热点后重跑")
        sys.exit(2)
    print(f"connected {host}")

    if not args.skip_push:
        push(c)

    data = run_bench(c)
    (ROOT / "bench_result.json").write_text(json.dumps(data, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    build_docx(data, DOCX)
    c.close()


if __name__ == "__main__":
    main()

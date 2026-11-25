"""Push the YOLO RKNN package to the NanoPi-M6 and run inference.

Usage:
    python deploy_yolo_board.py            # push + run samples
    python deploy_yolo_board.py --ping     # only check reachability
"""
from __future__ import annotations

import argparse
import posixpath
import sys
from pathlib import Path

import paramiko

HOSTS = ["10.18.113.103", "10.18.113.78"]
USER = "pi"
PASSWORDS = ["pi", "fa"]
LOCAL = Path(r"D:\projects\rk3588_yolo\deploy")
REMOTE = "~/yolo_rknn"


def connect(host: str, pwd: str) -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=USER, password=pwd, timeout=8,
              allow_agent=False, look_for_keys=False)
    return c


def find_board():
    for h in HOSTS:
        for p in PASSWORDS:
            try:
                c = connect(h, p)
                return c, h, p
            except Exception:
                continue
    return None, None, None


def run(c: paramiko.SSHClient, cmd: str) -> None:
    _, out, err = c.exec_command(cmd, get_pty=False)
    print(out.read().decode(errors="ignore"), end="")
    e = err.read().decode(errors="ignore")
    if e.strip():
        print(e, end="")


def put_dir(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    run_ssh = None  # keep signature simple
    try:
        sftp.stat(remote)
    except IOError:
        sftp.mkdir(remote)
    for item in sorted(local.iterdir()):
        rpath = posixpath.join(remote, item.name)
        if item.is_dir():
            put_dir(sftp, item, rpath)
        else:
            sftp.put(str(item), rpath)
            print(f"  -> {rpath}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", action="store_true")
    args = ap.parse_args()

    c, host, pwd = find_board()
    if c is None:
        print("板子不可达：先开机并连上手机热点（10.18.113.x），再重跑")
        sys.exit(2)
    print(f"connected {host} (pwd={pwd})")

    if args.ping:
        run(c, "hostname; uname -m; python3 -c 'import rknnlite;print(\"rknnlite ok\")'")
        return

    sftp = c.open_sftp()
    put_dir(sftp, LOCAL, REMOTE)
    sftp.close()

    run(c, f"cd {REMOTE} && ls -la && python3 -c 'import rknnlite, cv2, numpy; print(rknnlite.__file__)'")
    print("\n=== inference ===")
    run(c, f"cd {REMOTE} && python3 yolo_rknn_infer.py --model helmet_int8.rknn --source samples --bench 10")
    print("\n图片已存到板子 " + REMOTE + "/outputs/")


if __name__ == "__main__":
    main()

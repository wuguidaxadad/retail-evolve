import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--app-only", action="store_true")
args = parser.parse_args()
state = ROOT / ".runtime/services.json"
if state.exists():
    tracked = json.loads(state.read_text())
    for name in (
        ["service-worker", "worker", "api", "codex-gateway"]
        if args.app_only
        else ["service-worker", "worker", "api", "codex-gateway", "redis", "mysql"]
    ):
        item = tracked.get(name)
        if not item:
            continue
        pid = item["pid"]
        command = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True
        ).stdout
        cwd = subprocess.run(
            ["/usr/sbin/lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            capture_output=True,
            text=True,
        ).stdout
        signature = " ".join(item["command"][1:])
        if ("n" + str(ROOT)) not in cwd or signature not in command:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(80):
            status = subprocess.run(
                ["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True
            ).stdout.strip()
            if not status or status.startswith("Z"):
                break
            time.sleep(0.25)
    print("已向本项目的服务发送停止信号。")

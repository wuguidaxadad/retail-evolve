#!/usr/bin/env python3
"""Start isolated project services; never use or modify the user's other databases."""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if sys.prefix == sys.base_prefix:
    os.execv(str(ROOT / ".venv/bin/python"), [str(ROOT / ".venv/bin/python"), __file__])
import pymysql  # noqa: E402 - re-execute under the project venv before importing dependencies.
import redis  # noqa: E402

RUNTIME = ROOT / ".runtime"
LOGS = RUNTIME / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
STATE = RUNTIME / "services.json"
tracked = json.loads(STATE.read_text()) if STATE.exists() else {}


def persist():
    STATE.write_text(json.dumps(tracked, indent=2))


def occupied(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def alive(pid):
    try:
        os.kill(int(pid), 0)
        status = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True
        ).stdout.strip()
        return bool(status) and not status.startswith("Z")
    except (ProcessLookupError, ValueError):
        return False


def spawn(name, command):
    if tracked.get(name) and alive(tracked[name]["pid"]):
        return
    log = open(LOGS / f"{name}.log", "ab", buffering=0)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log,
        stderr=log,
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        start_new_session=True,
    )
    tracked[name] = {"pid": process.pid, "command": command}
    persist()


def wait_for(check, timeout=40):
    for _ in range(timeout * 4):
        try:
            if check():
                return
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError("启动超时，请查看 .runtime/logs")


mysql = shutil.which("mysqld") or next(
    (
        p
        for p in ["/usr/local/opt/mysql/bin/mysqld", "/opt/homebrew/opt/mysql/bin/mysqld"]
        if Path(p).exists()
    ),
    None,
)
if not mysql:
    raise SystemExit("未找到 MySQL。安装 MySQL 后重试，或使用 docker compose up --build。")
data = RUNTIME / "mysql"
data.mkdir(exist_ok=True)
if not (data / "mysql").exists():
    with open(LOGS / "mysql-init.log", "ab") as log:
        subprocess.run(
            [mysql, "--no-defaults", "--initialize-insecure", f"--datadir={data}"],
            stdout=log,
            stderr=log,
            check=True,
        )
if occupied(3307) and not (tracked.get("mysql") and alive(tracked["mysql"]["pid"])):
    raise SystemExit("3307 已由其他服务占用，未连接或修改该实例。")
spawn(
    "mysql",
    [
        mysql,
        "--no-defaults",
        f"--datadir={data}",
        "--port=3307",
        "--bind-address=127.0.0.1",
        "--mysqlx=0",
        f"--socket={RUNTIME}/mysql.sock",
        f"--pid-file={RUNTIME}/mysql.pid",
        f"--log-error={LOGS}/mysql-error.log",
    ],
)


def connection():
    return pymysql.connect(host="127.0.0.1", port=3307, user="root", password="", autocommit=True)


wait_for(lambda: connection().close() is None)
with connection() as conn:
    with conn.cursor() as c:
        c.execute("CREATE DATABASE IF NOT EXISTS retail_evolve CHARACTER SET utf8mb4")
        c.execute("CREATE USER IF NOT EXISTS 'retail'@'127.0.0.1' IDENTIFIED BY 'retail_local'")
        c.execute("GRANT ALL ON retail_evolve.* TO 'retail'@'127.0.0.1'")
redis_binary = RUNTIME / "redis-server"
if not redis_binary.exists():
    raise SystemExit("缺少项目 Redis，请先运行 scripts/install.py 或使用 Docker Compose。")
if occupied(6381) and not (tracked.get("redis") and alive(tracked["redis"]["pid"])):
    raise SystemExit("6381 已由其他服务占用，未修改该实例。")
(RUNTIME / "redis").mkdir(exist_ok=True)
spawn(
    "redis",
    [
        str(redis_binary),
        "--bind",
        "127.0.0.1",
        "--port",
        "6381",
        "--appendonly",
        "yes",
        "--dir",
        str(RUNTIME / "redis"),
        "--save",
        "60 1",
    ],
)
wait_for(
    lambda: redis.Redis(
        host="127.0.0.1",
        port=6381,
        socket_timeout=1,
        socket_connect_timeout=1,
        retry=redis.retry.Retry(redis.backoff.NoBackoff(), 0),
    ).ping()
)
subprocess.run([sys.executable, "-m", "app.bootstrap"], check=True)
if occupied(8765) and not (tracked.get("api") and alive(tracked["api"]["pid"])):
    raise SystemExit("8765 已由其他服务占用。")
spawn(
    "api", [sys.executable, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8765"]
)
if "--product-only" not in sys.argv:
    spawn("worker", [sys.executable, "-m", "app.worker"])
if occupied(8770) and not (tracked.get("codex-gateway") and alive(tracked["codex-gateway"]["pid"])):
    raise SystemExit("8770 已由其他服务占用。")
spawn(
    "codex-gateway",
    [
        sys.executable,
        "-m",
        "uvicorn",
        "app.codex_gateway:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8770",
    ],
)
wait_for(lambda: occupied(8765))
wait_for(lambda: occupied(8770))
print("Retail Evolve 已启动：http://127.0.0.1:8765")

spawn("service-worker", [str(ROOT / ".venv/bin/python"), "-m", "app.service_worker"])

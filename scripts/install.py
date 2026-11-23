"""Install Python dependencies and a checksum-pinned local Redis runtime."""

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if sys.version_info[:2] != (3, 12):
    raise SystemExit("请使用 Python 3.12。该固定版本的 τ 含 audioop，尚不支持 Python 3.13。")
python = ROOT / ".venv/bin/python"
if not python.exists():
    venv.EnvBuilder(with_pip=True).create(ROOT / ".venv")
subprocess.run([str(python), "-m", "pip", "install", "-r", "requirements.lock"], check=True)
subprocess.run(
    [str(python), "-m", "pip", "install", "--no-deps", "-e", "vendor/ace", "-e", "vendor/tau2"],
    check=True,
)
runtime = ROOT / ".runtime"
runtime.mkdir(exist_ok=True)
if not (runtime / "redis-server").exists():
    # Compilation requires the platform's standard C build tools.
    build = runtime / "build"
    build.mkdir(exist_ok=True)
    archive = build / "redis-7.4.2.tar.gz"
    urllib.request.urlretrieve("https://download.redis.io/releases/redis-7.4.2.tar.gz", archive)
    checksum = "4ddebbf09061cbb589011786febdb34f29767dd7f89dbe712d2b68e808af6a1f"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != checksum:
        raise SystemExit("Redis 下载校验失败")
    with tarfile.open(archive) as tar:
        tar.extractall(build, filter="data")
    source = build / "redis-7.4.2"
    subprocess.run(
        ["make", "-j4"], cwd=source, check=True, env={**os.environ, "LC_ALL": "C", "LANG": "C"}
    )
    for name in ["redis-server", "redis-cli"]:
        shutil.copy2(source / "src" / name, runtime / name)
print("依赖安装完成。填写 .env 后运行 .venv/bin/python scripts/start_local.py。")

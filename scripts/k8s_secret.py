"""Create a cluster-local Secret without printing or writing credentials."""
import json
import secrets
import subprocess
from pathlib import Path

from dotenv import dotenv_values

namespace = "retail-evolve"
existing = subprocess.run(
    ["kubectl", "-n", namespace, "get", "secret", "retail-secrets", "--ignore-not-found", "-o", "name"],
    capture_output=True,
    text=True,
    check=True,
)
if existing.stdout.strip():
    print("Existing retail-secrets preserved.")
else:
    values = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
    if not values.get("OPENAI_API_KEY") or not values.get("OPENAI_BASE_URL"):
        raise SystemExit("Configure OPENAI_API_KEY and OPENAI_BASE_URL in the local .env first.")
    password = secrets.token_hex(24)
    data = {
        "MYSQL_DATABASE": "retail_evolve",
        "MYSQL_USER": "retail",
        "MYSQL_PASSWORD": password,
        "MYSQL_ROOT_PASSWORD": secrets.token_hex(24),
        "MYSQL_URL": f"mysql+pymysql://retail:{password}@mysql:3306/retail_evolve?charset=utf8mb4",
        "OPENAI_API_KEY": values["OPENAI_API_KEY"],
        "OPENAI_BASE_URL": values["OPENAI_BASE_URL"],
    }
    manifest = {
        "apiVersion": "v1", "kind": "Secret",
        "metadata": {"name": "retail-secrets", "namespace": namespace},
        "type": "Opaque", "stringData": data,
    }
    subprocess.run(
        ["kubectl", "create", "-f", "-"], input=json.dumps(manifest), text=True, check=True
    )

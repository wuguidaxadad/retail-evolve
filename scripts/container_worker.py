"""Wait for the API and local Responses gateway before consuming queued work."""
import os
import time
import urllib.request

urls = [
    os.getenv("API_READY_URL", "http://api:8765/api/ready"),
    os.getenv("CODEX_GATEWAY_URL", "http://127.0.0.1:8770/v1").removesuffix("/v1") + "/health",
]
for url in urls:
    for attempt in range(120):
        try:
            with urllib.request.urlopen(url, timeout=3):
                break
        except (OSError, TimeoutError):
            time.sleep(2)
    else:
        raise RuntimeError("Dependency did not become ready: " + url)
os.execvp("python", ["python", "-m", "app.worker"])

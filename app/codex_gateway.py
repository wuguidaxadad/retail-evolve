"""Responses compatibility diagnostics for the configured provider (loopback only)."""

from __future__ import annotations

import json
import secrets

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from .config import ROOT, settings
from .native_bridge import sanitize

app = FastAPI()


@app.get("/health")
async def health():
    return {"alive": True}


@app.post("/v1/responses")
async def responses(request: Request):
    if not secrets.compare_digest(
        request.headers.get("authorization", ""), "Bearer " + settings.api_key
    ):
        raise HTTPException(401, "Invalid local model key")
    body = await request.json()
    path = ROOT / ".runtime/codex-wire-request.json"
    path.write_text(json.dumps(sanitize(body), ensure_ascii=False, indent=2))
    # Verified A/B: a6 rejected the Codex/GPT-5 identity sentence without using
    # tools, while the same CLI request with generic identity edited and tested a file.
    # Preserve all remaining capabilities/rules; benchmark prompts are untouched.
    instructions = body.get("instructions", "")
    if isinstance(instructions, str) and instructions.startswith("You are Codex"):
        first, separator, rest = instructions.partition("\n")
        body["instructions"] = (
            "You are a software engineering assistant with the supplied tools.\n" + rest
        )
    headers = {"Authorization": "Bearer " + settings.api_key, "Content-Type": "application/json"}
    client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=15))
    upstream = await client.send(
        client.build_request("POST", settings.base_url + "/responses", headers=headers, json=body),
        stream=True,
    )
    if not body.get("stream"):
        data = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        return Response(data, status_code=upstream.status_code, media_type="application/json")

    async def stream():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        stream(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )

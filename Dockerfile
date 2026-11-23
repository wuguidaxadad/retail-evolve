FROM node:22-bookworm-slim AS codex
ARG CODEX_VERSION=0.144.6
RUN npm install -g @openai/codex@${CODEX_VERSION}

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates bubblewrap \
    && rm -rf /var/lib/apt/lists/*
COPY --from=codex /usr/local/bin/node /usr/local/bin/node
COPY --from=codex /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/@openai/codex/bin/codex.js /usr/local/bin/codex \
    && codex --version
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY vendor ./vendor
RUN pip install --no-cache-dir --no-deps -e vendor/ace -e vendor/tau2
COPY app ./app
COPY web ./web
COPY scripts ./scripts
COPY UPSTREAM.json .
RUN git clone https://github.com/neosigmaai/auto-harness /opt/auto-harness \
    && git -C /opt/auto-harness checkout 52360dbbeb8cea8886d90778e634a145c51da735 \
    && git config --system --add safe.directory /opt/auto-harness \
    && useradd --create-home --uid 1000 app \
    && mkdir -p artifacts docs .runtime \
    && chown -R app:app /app
ENV AUTO_HARNESS_SOURCE=/opt/auto-harness PYTHONUNBUFFERED=1
USER app
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8765"]

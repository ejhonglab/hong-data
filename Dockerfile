# the commit the code built is of, and whether it differs from it (app/version.py), for the
# page's footer: only this stage sees the repository's .git
FROM python:3.12-slim AS version
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
 && git config --global --add safe.directory '*'
WORKDIR /src
COPY . .
RUN python -m app.version /version.json

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN useradd --system --uid 10001 --create-home olfdb
WORKDIR /srv/olfdb

COPY requirements.txt .
# git only to install chemutils (and its pubchemprops) from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get purge -y --auto-remove git && rm -rf /var/lib/apt/lists/*

COPY app ./app
COPY config ./config
COPY reference_data ./reference_data
COPY manage.py .
COPY --from=version /version.json ./VERSION.json

# uploaded data (SQLite DB + files) lives in a volume mounted here; the code stays root's,
# so the app cannot change it
ENV OLFDB_DATA_DIR=/data OLFDB_REFERENCE_DIR=/srv/olfdb/reference_data
RUN mkdir -p /data && chown -R olfdb:olfdb /data
VOLUME ["/data"]
USER olfdb
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "30"]

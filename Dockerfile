# the commit the code built is of, and whether it differs from it (app/version.py), for the
# page's footer: only this stage sees the repository's .git
FROM python:3.12-slim AS version
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
 && git config --global --add safe.directory '*'
WORKDIR /src
COPY . .
RUN python -m app.version /version.json

# what the web server and the fit worker share: Python and the web server's packages
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN useradd --system --uid 10001 --create-home olfdb
WORKDIR /srv/olfdb

COPY requirements.txt .
# git only to install chemutils (and its pubchemprops) and hong2p from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get purge -y --auto-remove git && rm -rf /var/lib/apt/lists/*

# the fit worker (fit_worker.py, docker-compose.yml's worker service): also al-diag-seg, PyTorch (with CUDA) and
# Cellpose-SAM's weights, which it would otherwise download where the container cannot write. Its packages are
# installed before the app is copied in, so a change to the app does not install them again.
FROM base AS worker
COPY requirements-worker.txt .
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && pip install --no-cache-dir -r requirements-worker.txt \
 && apt-get purge -y --auto-remove git && rm -rf /var/lib/apt/lists/*
ENV CELLPOSE_LOCAL_MODELS_PATH=/srv/olfdb/cellpose_models
# (downloaded as root, readable by root only: made readable by the user the worker runs as)
RUN python -c "from cellpose.models import cache_model_path; cache_model_path('cpsam_v2')" \
 && chmod -R a+rX "$CELLPOSE_LOCAL_MODELS_PATH"
# the Grabe et al. 2016 atlas, for fits that add it to their flies' atlas (fits.REFERENCES), as al_atlas exported it at
# this commit (https://github.com/ejhonglab/al_atlas: its labels and their names; checksums of them at that commit)
ARG AL_ATLAS=88ebada36bed414e7799d887b6fceca034bbc0cb
ADD --chmod=u=rwX,go=rX --checksum=sha256:5e0fa4bceb09e58832c84b9fe3b64ac742cf5beda429f8435733315b0d0cd815 \
    https://raw.githubusercontent.com/ejhonglab/al_atlas/${AL_ATLAS}/grabe2016/labels.tif al_atlas/grabe2016/
ADD --chmod=u=rwX,go=rX --checksum=sha256:4717da31369525b539d77bbfbb7f1834ac1fb921fc1464771e5528e7e82d253e \
    https://raw.githubusercontent.com/ejhonglab/al_atlas/${AL_ATLAS}/grabe2016/structures.csv al_atlas/grabe2016/
# al-diag-seg alone, after what it needs (above) and Cellpose's weights: a new one rebuilds only this
COPY requirements-alseg.txt .
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && pip install --no-cache-dir --no-deps -r requirements-alseg.txt && pip check \
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

CMD ["python", "-m", "app.fit_worker"]


# the web server (the default target, docker-compose.yml's olfdb service): with the X libraries
# RDKit's drawing loads (the Odors tab's structures), not in the slim image, nor in the worker's
FROM base AS web
RUN apt-get update && apt-get install -y --no-install-recommends libxrender1 libxext6 libexpat1 \
 && rm -rf /var/lib/apt/lists/*
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

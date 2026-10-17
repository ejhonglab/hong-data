# Hosting on an intranet Ubuntu server

The app is one container (Python/uvicorn, SQLite inside a bind-mounted `data/`
directory). Docker isolates it from the rest of the server, Docker's restart policy
brings it back if it crashes, and a small systemd unit starts the stack on boot and
gives you `systemctl` / `journalctl` control. Tested layout: code in `/opt/olfdb`,
served on port 8080.

## 1. Install Docker (once per server)

Ubuntu 26.04 ships Docker and the Compose plugin in its own repositories:

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"     # optional: run docker without sudo (re-login afterwards)
docker compose version               # should print v2.x
```

(If you prefer Docker's upstream packages, follow https://docs.docker.com/engine/install/ubuntu/
and install `docker-ce docker-compose-plugin` instead; everything below is the same.)

## 2. Put the code on the server

```bash
sudo mkdir -p /opt/olfdb
sudo chown "$USER" /opt/olfdb
# copy the project there (git clone, scp, rsync ...), e.g.:
rsync -a --exclude data/ ./olfdb/ server:/opt/olfdb/
cd /opt/olfdb
mkdir -p data
sudo chown -R 10001:10001 data        # the container runs as uid 10001 ("olfdb")
```

`data/` is the only thing that needs backing up: it holds `olfdb.sqlite3`, every
uploaded file (`data/datasets/<id>/...`) and the short-lived staging area.

Before the first start, have a look at:

- `docker-compose.yml`: host port (`8080:8000`), upload size limit.
- `config/glomeruli.txt`: glomerulus names accepted as ROI labels.
- `reference_data/manifest.json`: the fixed matrices shown in the "Reference matrices"
  tab (format in `reference_data/README.md`). Replace the two synthetic examples.

## 3. Build and start it

```bash
cd /opt/olfdb
docker compose up -d --build
curl -s http://localhost:8080/api/health     # {"ok":true,...}
docker compose logs -f                       # Ctrl-C to stop following
```

Open `http://<server>:8080/` from any machine on the intranet. The container has
`restart: unless-stopped`, so Docker already restarts it after a crash and after a
reboot (as long as the docker service is enabled).

## 4. Manage it with systemd (start on boot, restart on failure, logs in journald)

```bash
sudo cp deploy/olfdb.service /etc/systemd/system/olfdb.service
sudo systemctl daemon-reload
sudo systemctl enable --now olfdb
systemctl status olfdb
journalctl -u olfdb -f
```

The unit runs `docker compose up --build` in the foreground with `Restart=always`, so
systemd supervises the compose process while Docker supervises the container. To
deploy a new version:

```bash
cd /opt/olfdb && git pull        # or rsync the new files over
sudo systemctl restart olfdb     # rebuilds the image and recreates the container
```

If you installed the unit, do not also run `docker compose up -d` by hand; use
`systemctl start/stop/restart olfdb`.

## 5. Backups

Everything lives in `/opt/olfdb/data`. A consistent copy of the SQLite file is made
with the sqlite backup API; the dataset files can simply be copied:

```bash
# /etc/cron.daily/olfdb-backup  (chmod +x)
#!/bin/sh
set -e
DEST=/srv/backups/olfdb/$(date +%F)
mkdir -p "$DEST"
docker compose -f /opt/olfdb/docker-compose.yml exec -T olfdb \
  python -c "import sqlite3; sqlite3.connect('/data/olfdb.sqlite3').backup(sqlite3.connect('/data/backup.sqlite3'))"
mv /opt/olfdb/data/backup.sqlite3 "$DEST/olfdb.sqlite3"
rsync -a /opt/olfdb/data/datasets/ "$DEST/datasets/"
```

Restoring is the reverse: stop the service, put the files back under `data/`, start.

## 6. Optional: reverse proxy, TLS, access control

The app itself has no login (it is meant for a trusted intranet). If you want
hostname-based access, TLS, or HTTP basic auth, put nginx (or Caddy) in front:

```nginx
server {
    listen 80;
    server_name olfdb.lab.internal;
    client_max_body_size 1200m;          # uploads can be large
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_read_timeout 600s;
        proxy_set_header Host $host;
        # auth_basic "olfdb"; auth_basic_user_file /etc/nginx/olfdb.htpasswd;
    }
}
```

and restrict the port with the firewall, e.g. `sudo ufw allow from 10.0.0.0/8 to any port 8080`.

## 7. Day-to-day maintenance

```bash
docker compose exec olfdb python manage.py list            # datasets in the DB
docker compose exec olfdb python manage.py show 3
docker compose exec olfdb python manage.py export 3 /data/export-3   # files + long CSV
docker compose exec -it olfdb python manage.py delete 3    # asks for confirmation
docker compose exec olfdb python manage.py cleanup-staging
sqlite3 data/olfdb.sqlite3 '.tables'                       # direct SQL access (apt install sqlite3)
```

Logs: `journalctl -u olfdb` (with the systemd unit) or `docker compose logs`.

## 8. Without Docker (plain virtualenv + systemd)

```bash
sudo apt install -y python3-venv
sudo useradd --system --home /opt/olfdb --shell /usr/sbin/nologin olfdb
cd /opt/olfdb
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
sudo chown -R olfdb:olfdb /opt/olfdb
sudo cp deploy/olfdb-venv.service /etc/systemd/system/olfdb.service
sudo systemctl daemon-reload && sudo systemctl enable --now olfdb
```

Same behaviour (boot start, restart on crash), just no container isolation.

## Configuration reference

| variable | default | meaning |
|---|---|---|
| `OLFDB_DATA_DIR` | `./data` (`/data` in Docker) | SQLite DB, staging uploads, dataset files |
| `OLFDB_REFERENCE_DIR` | `./reference_data` | fixed matrices + `manifest.json` |
| `OLFDB_GLOMERULI_FILE` | `./config/glomeruli.txt` | accepted glomerulus names (falls back to the built-in list if missing) |
| `OLFDB_MAX_UPLOAD_MB` | `1024` | reject data files larger than this |
| `OLFDB_STAGE_TTL_HOURS` | `48` | staged-but-never-saved uploads are deleted after this |

Keep `--workers 1`: SQLite handles concurrent readers fine, and one process avoids
write contention; the viewer work happens in the browser anyway.

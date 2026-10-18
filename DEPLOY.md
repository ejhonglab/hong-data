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
uploaded file (`data/datasets/<id>/...`), the odor table and its sync history, the
folders set in the Admin tab (`data/admin.json`), the short-lived staging area,
downloads being sent (`data/exports/`, deleted once sent), cached odor structure images
(`data/odor_structures/`) and temporary files (`data/tmp/`).

Before the first start, have a look at:

- `docker-compose.yml`: host port (`8080:8000`), upload size limit, odor sheet link
  (see "Odor sheet" below), where tensor is mounted (see "Tensor" below), and any other
  folder it may use (see "Folders it may read and write" below).
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

### Odor sheet

The Odors tab keeps the app's own table of odor metadata (one row per chemical), which
it syncs from a Google sheet when someone asks it to, after showing what would change.
To set the sheet up:

1. Make a Google sheet with the columns of `config/odor_sheet_template.csv` (File →
   Import that file to start from the odors in the thesis tables). Column names are
   matched loosely; `aliases` are `;`-separated.
2. Share it as "anyone with the link can view" (the server reads its CSV export; no
   Google account or credentials are involved, and nothing is written to the sheet).
3. Put its link (as copied from the browser while editing it; include `#gid=...` if the
   odors are not on the first tab) in `OLFDB_ODOR_SHEET_URL` in `docker-compose.yml`, and
   restart: `docker compose up -d`.

The server needs outbound HTTPS to docs.google.com for the sheet, and to
pubchem.ncbi.nlm.nih.gov for structure images and InChIs (fetched once per CID, then
cached; InChIs in the database, a hundred CIDs per request).
Without a sheet, the Odors tab can still import a CSV in the same format.

### Tensor

Users give their data folders (Settings tab) relative to the top of the lab NAS,
tensor, which the server needs to be able to read: it checks that the folders exist,
and looks in them for the flies of each upload (raw data folder, ThorImage
`Experiment.xml`, odor YAMLs). Mount tensor on the server (by default the app expects it
at `/mnt/tank0/tensor`; `OLFDB_TENSOR_ROOT` sets another place). `docker-compose.yml`
bind-mounts that path into the container read-only, at the same path; change both
sides of that line, and `OLFDB_TENSOR_ROOT`, if it is elsewhere. The container runs as
uid 10001, so the files need to be readable by it (world-readable is enough); the
Settings tab says when a folder exists but cannot be read. If the path does not exist
on the host when the container starts, Docker creates it empty, and the Settings tab
reports tensor as not mounted.

### Folders it may read and write

What the app may touch on the server is limited in two layers:

1. **What it can reach at all**, set on the server, where the app cannot change it: the
   container sees only what `docker-compose.yml` mounts (tensor read-only, `data/`
   writable), runs as an unprivileged user without capabilities, and cannot change its
   own image (`read_only: true`). For tensor itself, the strongest guarantee is the
   NAS's own: mount it read-only, or as a user that may only read.
2. **What it may use of that**, set in the Admin tab (to the right of Docs) and saved in
   `data/admin.json`: folders it may read (by default `OLFDB_TENSOR_ROOT`) and folders it
   may also write (none by default; nothing writes there yet: they are meant for
   backups). Its data folder is always readable and writable, and it writes nowhere
   else (temporary files go to `data/tmp/`). Both lists are fixed when the server
   starts: the app checks the paths it reads on tensor against them, and has the kernel
   deny it everything else, through
   [Landlock](https://docs.kernel.org/userspace-api/landlock.html) (Linux 5.13 or later;
   Docker's default seccomp profile allows it). A running process cannot lift that, so
   changes take effect when the server restarts, which the Admin tab can do: the server
   exits, and Docker's restart policy (or systemd, with `deploy/olfdb-venv.service`)
   starts it again. The Admin tab says whether Landlock is in use; `OLFDB_LANDLOCK=0`
   turns it off, leaving only the app's own checks.

So to let it write somewhere, e.g. backups on tensor: mount that folder writable, as in
the commented-out line in `docker-compose.yml` (on tensor, the folder has to exist
already), `docker compose up -d`, then add it in the Admin tab and restart from there. A
folder only added in the Admin tab gets a warning there: not on the server (not mounted
into the container), or not writable (mounted read-only).

The Admin tab first asks whether you are Tom, Jon or Betty, or have asked one of them: a
reminder, not a lock. Anyone who can open the site can change these settings and
restart it; to require a password, see section 6.

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
`systemctl start/stop/restart olfdb`. A restart from the Admin tab restarts the
container as it is, without picking up new code.

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
    location /api/admin {                # the Admin tab: changing its settings, restarting
        auth_basic "olfdb admin"; auth_basic_user_file /etc/nginx/olfdb-admin.htpasswd;
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
    }
}
```

(`sudo htpasswd -c /etc/nginx/olfdb-admin.htpasswd tom`, from `apache2-utils`, makes the
password file for the Admin tab.)

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
python3 -m venv venv && venv/bin/pip install -r requirements.txt
sudo chown -R olfdb:olfdb /opt/olfdb
sudo cp deploy/olfdb-venv.service /etc/systemd/system/olfdb.service
sudo systemctl daemon-reload && sudo systemctl enable --now olfdb
```

Same behaviour (boot start, restart on crash or from the Admin tab). Instead of a
container, the unit's sandboxing limits what the server can reach: every file read-only
but `data/`, no home folders, and of `/mnt` only tensor, read-only (see "Folders it may
read and write"). A folder it may write needs `ReadWritePaths=` there too (and
`BindPaths=` if it is under `/mnt`).

## Configuration reference

| variable | default | meaning |
|---|---|---|
| `OLFDB_DATA_DIR` | `./data` (`/data` in Docker) | SQLite DB, staging uploads, dataset files |
| `OLFDB_REFERENCE_DIR` | `./reference_data` | fixed matrices + `manifest.json` |
| `OLFDB_GLOMERULI_FILE` | `./config/glomeruli.txt` | accepted glomerulus names (falls back to the built-in list if missing) |
| `OLFDB_MAX_UPLOAD_MB` | `1024` | reject data files larger than this |
| `OLFDB_STAGE_TTL_HOURS` | `48` | staged-but-never-saved uploads are deleted after this |
| `OLFDB_ODOR_SHEET_URL` | (none) | Google sheet the odor table is synced from (see "Odor sheet") |
| `OLFDB_ODOR_SHEET_CHECK_MINUTES` | `10` | how often (at most) the sheet is read to tell whether it changed |
| `OLFDB_TENSOR_ROOT` | `/mnt/tank0/tensor` | the lab NAS as mounted on the server (see "Tensor"); users' folders are relative to it |
| `OLFDB_LANDLOCK` | `1` | `0` to stop having the kernel (Landlock) hold the server to the folders set in the Admin tab (see "Folders it may read and write") |

Keep `--workers 1`: SQLite handles concurrent readers fine, and one process avoids
write contention; the viewer work happens in the browser anyway.

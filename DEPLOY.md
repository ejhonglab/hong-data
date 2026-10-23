# Hosting on an intranet Ubuntu server

The app is one container (Python/uvicorn, SQLite inside a bind-mounted `data/`
directory). Docker isolates it from the rest of the server, Docker's restart policy
brings it back if it crashes, and a small systemd unit starts the stack on boot and
gives you `systemctl` / `journalctl` control. Tested layout: code in `/opt/olfdb`,
served on port 8080.

## 1. Install Docker (once per server)

If the server already has Docker, check it instead, and go on to step 2 if both print a
version:

```bash
docker compose version     # v2.x or later (v5.x is fine)
docker buildx version      # Compose v5 builds the image with buildx
```

Compose went from v2 to v5 (skipping 3 and 4), and v5 works the same here. Only v1, the
separate `docker-compose` command, is too old.

Otherwise, Ubuntu 26.04 ships Docker, Compose and buildx in its own repositories:

```bash
sudo apt update
sudo apt install docker.io docker-compose-v2 docker-buildx
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"     # optional: run docker without sudo (re-login afterwards)
```

(If you prefer Docker's upstream packages, follow https://docs.docker.com/engine/install/ubuntu/
and install `docker-ce docker-buildx-plugin docker-compose-plugin` instead; everything below
is the same.) Use one set or the other, never both: they install the same files, so apt
fails with "trying to overwrite '/usr/libexec/docker/cli-plugins/…'" (or offers to remove
Docker) when the other set is already there.

## 2. Put the code on the server

```bash
sudo mkdir -p /opt/olfdb
sudo chown "$USER" /opt/olfdb
# the project goes in /opt/olfdb itself, not a folder inside it (the systemd units below
# expect /opt/olfdb/docker-compose.yml):
git clone https://github.com/ejhonglab/hong-data.git /opt/olfdb
# (or from another computer: rsync -a --exclude data/ ./hong-data/ server:/opt/olfdb/)
cd /opt/olfdb
mkdir -p data
sudo chown -R 10001:10001 data        # the container runs as uid 10001 ("olfdb")
```

`data/` is the only thing that needs backing up: it holds `olfdb.sqlite3`, every
uploaded file (`data/datasets/<id>/...`; those of datasets deleted in the Admin tab in
`data/deleted/<id>/`), the odor table and its sync history, the
folders set in the Admin tab (`data/admin.json`), the short-lived staging area,
downloads being sent (`data/exports/`, deleted once sent), copies of the reference
datasets added in the Admin tab (`data/references/`; their files stay where they are, to
be backed up there) and temporary files (`data/tmp/`).

Before the first start, have a look at:

- `docker-compose.yml`: host port (`8080:8000`), upload size limit, where tensor is
  mounted (see "Tensor" below), and any other folder it may use (see "Folders it may
  read and write" below). It also mounts the server's `/etc/hostname`, for the Admin
  tab to show the server's name (the container's own is random).
- `config/glomeruli.txt`: glomerulus names accepted as ROI labels.
- `reference_data/manifest.json`: the built-in matrices shown in the "Reference
  matrices" tab (format in `reference_data/README.md`). Replace the two synthetic
  examples, or add matrices in the Admin tab instead. A file added there from outside
  the folders the server may read also has to be within its reach: in Docker, a
  read-only volume in `docker-compose.yml` (`- /srv/refs:/srv/refs:ro`, then
  `docker compose up -d`); with systemd, outside home folders, `/tmp`, `/mnt` and
  `/local` (or with a `BindReadOnlyPaths=` for it). The Admin tab says which, when it is
  added.

## 3. Build and start it

```bash
cd /opt/olfdb
docker compose up -d --build
curl -s http://localhost:8080/api/health     # {"ok":true,...}
docker compose logs -f                       # Ctrl-C to stop following
```

Go on only once the health check answers `{"ok":true,...}`: the systemd unit in section
4 runs the same command, so whatever stops it here stops the unit too. Run `docker
compose` from `/opt/olfdb` itself (as the unit does): the Admin tab takes the folder it
is run from to be where the container's files are on the server, and says so when that
does not hold its data folder.

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
   odors are not on the first tab) under "Odor sheet" in the Admin tab, and save: it is
   used at once, and read right away to check that it can be. (`OLFDB_ODOR_SHEET_URL` on
   the server also works, while no link is saved in the Admin tab.)

The server needs outbound HTTPS to docs.google.com for the sheet, and to
pubchem.ncbi.nlm.nih.gov for the InChIs of the odors' CIDs (fetched once per CID, a
hundred per request, and kept in the database), from which their structures are drawn
on the server.
Without a sheet, the Admin tab can still import a CSV in the same format.

### Tensor

Users give their data folders (Settings tab) relative to the top of the lab NAS,
tensor, which the server needs to be able to read: it checks that the folders exist,
and looks in them for the flies of each upload (raw data folder, ThorImage
`Experiment.xml`, odor YAMLs). Mount tensor on the server (by default the app expects it
at `/local/tensor`; `OLFDB_TENSOR_ROOT` sets another place). `docker-compose.yml`
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
   may also write (none by default; backups made from the Admin tab go into one of
   them). Its data folder is always readable and writable, and it writes nowhere
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
folder only added in the Admin tab gets a warning there: not inside the container, or
not writable (mounted read-only), with the line to add to `docker-compose.yml` (with
the systemd unit: to the unit). Restarting from the Admin tab does not mount anything:
`docker compose up -d` makes the container again with the new line.

To add folders from the Admin tab alone later, mount a folder holding them once, e.g.
`- /local/tensor-archive:/local/tensor-archive` (writable): any folder in it can then
be added in the Admin tab and used after a restart from there. The Admin tab's lists
(and Landlock) still decide what the server uses, but the container itself can then
reach all of that folder, so mount no more than is needed.

The Admin tab first asks whether you are Tom, Jon or Betty, or have asked one of them: a
reminder, not a lock. Anyone who can open the site can change these settings, restart
it and delete datasets; to require a password, see section 6.

## 4. Manage it with systemd (start on boot, restart on failure, logs in journald)

```bash
sudo cp deploy/olfdb.service /etc/systemd/system/olfdb.service
sudo systemctl daemon-reload
sudo systemctl enable --now olfdb
systemctl status olfdb     # "active (running)", and still a minute later
journalctl -u olfdb -f
```

If it fails again and again, the unit gives up after 5 starts in 10 minutes, and
`systemctl status olfdb` shows it as failed. `journalctl -u olfdb -n 100` says why
("Assertion failed" if the code is not in `/opt/olfdb` itself). Once that is fixed:
`sudo systemctl reset-failed olfdb && sudo systemctl start olfdb`.

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

Everything lives in `/opt/olfdb/data`. The Admin tab can back it up (*Backups*) into a
folder it may write: mount one writable (the commented-out line in
`docker-compose.yml`), add it in the Admin tab under *May also write*, and restart from
there. Each backup is a folder `olfdb-backup-<date>_<time>` laid out like `data/` (but
for uploads under way, downloads and temporary files), with files unchanged since the
one before hard-linked to it, and a `backup.json` of its own: the list of what it holds
(each file's size and modification time, and a fingerprint of the database's
contents), which the Admin tab compares with the data to tell whether the backup is
current. To restore one: stop the server, copy its contents into `/opt/olfdb/data`
(all but `backup.json`, which is not part of the data), fix their owner for the
container (`sudo chown -R 10001:10001 data`), start it.

Without the Admin tab, e.g. from cron: a consistent copy of the SQLite file is made
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
cp /opt/olfdb/data/admin.json "$DEST/" 2>/dev/null || true   # folders set in the Admin tab, if any
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
    location /api/admin {                # the Admin tab: changing its settings, restarting, deleting datasets
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
docker compose exec -it olfdb python manage.py delete 3    # asks for confirmation; erases its files
                                                           # (the Admin tab moves them to data/deleted/)
docker compose exec olfdb python manage.py cleanup-staging
sqlite3 data/olfdb.sqlite3 '.tables'                       # direct SQL access (apt install sqlite3)
```

Logs: `journalctl -u olfdb` (with the systemd unit) or `docker compose logs`.

## 8. Without Docker (plain virtualenv + systemd)

```bash
sudo apt install python3-venv
sudo useradd --system --home /opt/olfdb --shell /usr/sbin/nologin olfdb
cd /opt/olfdb
python3 -m venv venv && venv/bin/pip install -r requirements.txt
mkdir -p data
sudo chown -R olfdb:olfdb /opt/olfdb
# a first run in the foreground, as the service runs it: once
# `curl -s http://localhost:8080/api/health` (in another terminal) answers {"ok":true,...},
# stop it with Ctrl-C
sudo -u olfdb env OLFDB_DATA_DIR=/opt/olfdb/data venv/bin/uvicorn app.main:app --port 8080
sudo cp deploy/olfdb-venv.service /etc/systemd/system/olfdb.service
sudo systemctl daemon-reload && sudo systemctl enable --now olfdb
systemctl status olfdb     # "active (running)", and still a minute later
```

Same behaviour (boot start, restart on crash or from the Admin tab; giving up after 5
starts in 2 minutes, restarts from the Admin tab included, as in section 4). Instead of a
container, the unit's sandboxing limits what the server can reach: every file read-only
but `data/`, no home folders, and of `/mnt` and `/local` only tensor, read-only (see
"Folders it may read and write"). A folder it may write needs `ReadWritePaths=` there too
(and `BindPaths=` if it is under `/mnt` or `/local`).

## Configuration reference

| variable | default | meaning |
|---|---|---|
| `OLFDB_DATA_DIR` | `./data` (`/data` in Docker) | SQLite DB, staging uploads, dataset files |
| `OLFDB_REFERENCE_DIR` | `./reference_data` | fixed matrices + `manifest.json` |
| `OLFDB_GLOMERULI_FILE` | `./config/glomeruli.txt` | accepted glomerulus names (falls back to the built-in list if missing) |
| `OLFDB_MAX_UPLOAD_MB` | `1024` | reject data files larger than this |
| `OLFDB_STAGE_TTL_HOURS` | `48` | staged-but-never-saved uploads are deleted this long after they were last worked on |
| `OLFDB_ODOR_SHEET_URL` | (none) | Google sheet the odor table is synced from, while none is set in the Admin tab (see "Odor sheet") |
| `OLFDB_ODOR_SHEET_CHECK_MINUTES` | `10` | how often (at most) the sheet is read to tell whether it changed |
| `OLFDB_TENSOR_ROOT` | `/local/tensor` | the lab NAS as mounted on the server (see "Tensor"); users' folders are relative to it |
| `OLFDB_HOST_DIR` | the folder `docker compose` is run from (in `docker-compose.yml`) | Docker only: the folder of `docker-compose.yml` on the server, for the Admin tab to give paths on the server rather than in the container |
| `OLFDB_LANDLOCK` | `1` | `0` to stop having the kernel (Landlock) hold the server to the folders set in the Admin tab (see "Folders it may read and write") |

Keep `--workers 1`: SQLite handles concurrent readers fine, and one process avoids
write contention; the viewer work happens in the browser anyway.

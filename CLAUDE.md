# Notes for Claude

- If queued fits or predictions never start, don't trust `docker ps` ("Up"): check
  `docker inspect -f '{{.State.Status}} {{.State.Error}}' olfdb-worker` (and `olfdb`).
  After `restart-olfdb` it has been left `created` with "endpoint with name olfdb-worker
  already exists in network olfdb_default". Restarting again doesn't fix it; run
  `docker network disconnect -f olfdb_default olfdb-worker && docker start olfdb-worker`.
  See DEPLOY.md, section 7.

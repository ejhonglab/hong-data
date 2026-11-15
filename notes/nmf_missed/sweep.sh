#!/bin/sh
# The try-out (run.py) at each setting design.md gives numbers for, and what its candidates lie on (score.py's FINAL rows).
# S: a scratch folder with cache/ (cache.py's) and out/; run from this folder, on the server (the fit worker's image, its GPU).
S=${S:?a scratch folder with cache/ and out/}
HERE=$(dirname "$(readlink -f "$0")")
run() {
  tag=$1; shift
  docker run --rm --init --name olfdb-nmf-run --gpus device=0 --memory 16g --memory-swap 16g --pids-limit 512 --cpus 6 -e OMP_NUM_THREADS=6 -e OPENBLAS_NUM_THREADS=6 -e MKL_NUM_THREADS=6 --entrypoint python -v $HERE:/code:ro -v $S/cache:/cache:ro -v $S/out:/out -v /opt/olfdb/data/fits/7:/fit:ro olfdb-worker:latest /code/run.py $tag "$@" > $S/out/$tag.log 2>&1
  echo "== $tag $*"; python3 $HERE/score.py $S/out/$tag/rows.json -s | sed -n 3,5p
}
run base
run theta25 theta=0.25
run theta50 theta=0.5
run free15 n_free=15
run free40 n_free=40
run seed1 seed_um=1.0
run seed25 seed_um=2.5
run sig0 sigma_um=0.0
run sig1 sigma_um=1.0
run free0 nonneg=0
run on1 on=1
run it60 iters=60
run win7 win_um=7.0
run win14 win_um=14.0
run nonorm norm=none
run gap3 seed_gap_um=3.0

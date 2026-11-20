"""Movies held in the web server's memory for the movie player's plot of ROIs' responses (P,
`roi_responses`), so that each ROI plotted takes a fraction of a second, not seconds of reading
tensor.

Each player open on a movie with the fly's own ROIs asks for the movies a plot of its ROIs would
read (`roi_responses.recordings`: the fly's recordings registered together, else the movie alone)
to be read whole into memory, in the background: a *slot*, one per player (``holder``), at most
`slots` of them on the server (set in the Admin tab: `SETTING`, `DEFAULT_SLOTS` if not set). A
player moving to another fly's movie moves its slot there. Opening a movie when every slot is
taken holds nothing (its plots read tensor, as before); the first plot does take one, from the
player whose slot was used least recently (one closed first). A player closed gives its slot up
`LINGER_S` later, unless opened again by then.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import movies

SETTING = "movie_memory_slots"  # how many players may have movies held at once (Admin tab)
DEFAULT_SLOTS = 2
MAX_SLOTS = 16
LINGER_S = 30.0  # a closed player's movies are kept this long, in case it is opened again
CHUNK = 64 << 20  # bytes read at once


class Slot:
    def __init__(self, holder: str, fly: Path, key: str, paths: list[Path], who: Optional[str]):
        self.holder, self.fly, self.key, self.paths, self.who = holder, fly, key, paths, who
        self.arrays: dict[str, np.ndarray] = {}
        self.total = 0  # bytes to read (known as each movie is opened)
        self.read = 0
        self.state = "loading"  # then "ready", or "failed" (``error``)
        self.error: Optional[str] = None
        self.used = time.monotonic()
        self.released: Optional[float] = None  # when its player was closed
        self.dropped = False

    def info(self, now: float) -> dict:
        return {"holder": self.holder[:8], "who": self.who, "fly": str(self.fly), "movies": len(self.paths), "state": self.state,
                "error": self.error, "bytes": self.total, "read": self.read, "idle_s": round(now - self.used, 1),
                "frees_in_s": None if self.released is None else max(0.0, round(self.released + LINGER_S - now, 1))}


_lock = threading.Lock()
_slots: dict[str, Slot] = {}  # by holder


def _sweep(now: float) -> None:
    """Drops the slots of players closed more than `LINGER_S` ago (`_lock` held)."""
    for h, s in list(_slots.items()):
        if s.released is not None and now - s.released >= LINGER_S:
            _drop(h)


def _drop(holder: str) -> None:
    s = _slots.pop(holder, None)
    if s is not None:
        s.dropped = True
        s.arrays = {}  # (freed once no plot is reading them)


def _lru() -> Optional[str]:
    """The slot to give up first: of a closed player, else the one used least recently."""
    if not _slots:
        return None
    return min(_slots, key=lambda h: (_slots[h].released is None, _slots[h].used))


def hold(holder: str, fly: Path, paths: list[Path], slots: int, evict: bool = False, who: Optional[str] = None) -> Optional[dict]:
    """Has `paths` (resolved: the movies of `fly`, as the player names its folder) read into memory for `holder` (a player), in
    the background, in its slot (moved from another fly's, if it had one; a new one if fewer than
    `slots` are taken, else, with `evict`, the one `_lru` gives). Returns its slot's state
    (`Slot.info`), None if no slot was free."""
    now, key = time.monotonic(), fly_key(fly, paths)
    with _lock:
        _sweep(now)
        s = _slots.get(holder)
        if s is not None and s.key == key and s.state != "failed":
            s.released, s.used = None, now
            return s.info(now)
        if s is not None:
            _drop(holder)
        while len(_slots) >= max(0, slots):
            if not evict or not _slots:
                return None
            _drop(_lru())
        if slots <= 0:
            return None
        s = _slots[holder] = Slot(holder, fly, key, paths, who)
    threading.Thread(target=_load, args=(s,), name=f"movie-memory-{holder[:8]}", daemon=True).start()
    return s.info(now)


def _load(s: Slot) -> None:
    try:
        opened = []
        for p in s.paths:
            m = movies.open_movie(p)
            if isinstance(m.data, np.memmap) and m.data.flags.c_contiguous:  # (a TIFF in pieces is not held)
                opened.append(m)
                s.total += m.data.nbytes
        for m in opened:
            d = m.data
            out = np.empty(d.shape, dtype=d.dtype)
            view = memoryview(out.reshape(-1).view(np.uint8))
            with open(m.path, "rb", buffering=0) as f:
                f.seek(d.offset)
                done = 0
                while done < len(view):
                    if s.dropped:
                        return
                    n = f.readinto(view[done:done + CHUNK])
                    if not n:
                        raise OSError(f"{m.path.name} is shorter than its header says")
                    done += n
                    s.read += n
            with _lock:
                if s.dropped:
                    return
                s.arrays[str(m.path)] = out
        s.state = "ready"
    except (OSError, movies.MovieError) as e:
        s.state, s.error, s.arrays = "failed", str(e), {}


def array(path: Path) -> Optional[np.ndarray]:
    """A movie's data (as its `movies.Movie.data`, in memory), if a slot holds it; that slot then counts as used."""
    key = str(path)
    with _lock:
        for s in _slots.values():
            a = s.arrays.get(key)
            if a is not None:
                s.used = time.monotonic()
                return a
    return None


def release(holder: str) -> None:
    """A player closed: its slot is given up `LINGER_S` from now, unless it holds again by then."""
    with _lock:
        s = _slots.get(holder)
        if s is not None and s.released is None:
            s.released = time.monotonic()
    threading.Timer(LINGER_S + 0.5, sweep).start()


def sweep() -> None:
    with _lock:
        _sweep(time.monotonic())


def limit(slots: int) -> None:
    """Fewer slots allowed: those used least recently given up, down to `slots`."""
    with _lock:
        while len(_slots) > max(0, slots):
            _drop(_lru())


def status() -> list[dict]:
    """Each slot's state (`Slot.info`), most recently used first."""
    now = time.monotonic()
    with _lock:
        _sweep(now)
        return [s.info(now) for s in sorted(_slots.values(), key=lambda s: -s.used)]


def fly_key(fly: Path, paths: list[Path]) -> str:
    """What a slot is of: the fly's folder (as the player names it) and its movies, by their sizes and times."""
    def stamp(p: Path) -> str:
        try:
            st = os.stat(p)
            return f"{p}:{st.st_size}:{st.st_mtime_ns}"
        except OSError:
            return str(p)

    return "|".join([str(fly), *map(stamp, paths)])

// What the movie player (player.js) edits a prediction's ROIs with: their outlines simplified,
// their points found, added and deleted as ImageJ's polygon selections are, and the changes
// kept to undo and redo. Outlines are closed polygons, lists of [x, y] in pixels (as ImageJ's:
// from the corner of the image), the last point joined to the first.

const dist2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

// the squared distance from point p to the segment from a to b
function segDist2(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], len = dx * dx + dy * dy;
  const t = len ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len)) : 0;
  return (p[0] - a[0] - t * dx) ** 2 + (p[1] - a[1] - t * dy) ** 2;
}

// An outline with fewer points, none of those left out further than `tolerance` (pixels) from it
// (Ramer–Douglas–Peucker, as for a line, between its leftmost point and the one farthest from
// that, each way round): what is kept are points of the outline itself, so nothing moves but the
// corners cut. Never fewer than three points (of an outline with as many); with no tolerance, as
// it is (but for points repeated).
export function simplify(points, tolerance) {
  const pts = points.filter((p, i) => dist2(p, points[(i + 1) % points.length]) > 0 || points.length === 1);
  const n = pts.length, copy = (ps) => ps.map((p) => [p[0], p[1]]);
  if (n <= 3 || !(tolerance > 0)) return copy(pts);
  let a = 0, b = 0, far = -1;
  for (let i = 1; i < n; i++) if (pts[i][0] < pts[a][0] || (pts[i][0] === pts[a][0] && pts[i][1] < pts[a][1])) a = i;
  for (let i = 0; i < n; i++) { const d = dist2(pts[i], pts[a]); if (d > far) { far = d; b = i; } }
  const keep = new Uint8Array(n), tol2 = tolerance * tolerance;
  keep[a] = keep[b] = 1;
  // (each stretch from i round to j: the point of it farthest from the line between them kept, if further than the tolerance)
  const todo = [[a, b], [b, a]];
  let worst = -1, worstAt = -1; // (the point farthest from the line of the first two, should they be all that is kept)
  while (todo.length) {
    const [i, j] = todo.pop(), len = (j - i + n) % n || n;
    let at = -1, most = 0;
    for (let s = 1; s < len; s++) {
      const k = (i + s) % n, d = segDist2(pts[k], pts[i], pts[j]);
      if (d > most) { most = d; at = k; }
    }
    if (at >= 0 && most > worst && ((i === a && j === b) || (i === b && j === a))) { worst = most; worstAt = at; }
    if (at >= 0 && most > tol2) { keep[at] = 1; todo.push([i, at], [at, j]); }
  }
  if (keep.reduce((s, k) => s + k, 0) < 3 && worstAt >= 0) keep[worstAt] = 1;
  return copy(pts.filter((_, i) => keep[i]));
}

// The outline with point i made two, as ImageJ does when a point of a polygon is Shift-clicked:
// one a third of the way back to the point before it, one a third of the way on to the next.
export function splitPoint(points, i) {
  const n = points.length, p = points[i], before = points[(i - 1 + n) % n], after = points[(i + 1) % n];
  const part = (from, to, f) => [from[0] + (to[0] - from[0]) * f, from[1] + (to[1] - from[1]) * f];
  return [...points.slice(0, i), part(before, p, 2 / 3), part(p, after, 1 / 3), ...points.slice(i + 1)];
}

// The outline without point i (ImageJ's Alt-click), or null if that would leave fewer than three.
export function deletePoint(points, i) {
  return points.length > 3 ? [...points.slice(0, i), ...points.slice(i + 1)] : null;
}

// the point of the outline nearest (x, y), if within `reach` of it: its index, else -1
export function nearestPoint(points, x, y, reach) {
  let at = -1, least = reach * reach;
  points.forEach((p, i) => { const d = dist2(p, [x, y]); if (d <= least) { least = d; at = i; } });
  return at;
}

// whether (x, y) is inside the outline (even-odd)
export function inside(points, x, y) {
  let within = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i], [xj, yj] = points[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) within = !within;
  }
  return within;
}

// the outline's area, in square pixels
export function area(points) {
  let s = 0;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) s += points[j][0] * points[i][1] - points[i][0] * points[j][1];
  return Math.abs(s) / 2;
}

// States to undo and redo through: the one shown (`current`), those before it and, until a new
// one is made, those undone.
export class History {
  constructor(state, limit = 300) {
    this.states = [state];
    this.at = 0;
    this.limit = limit;
  }

  get current() { return this.states[this.at]; }

  get canUndo() { return this.at > 0; }

  get canRedo() { return this.at < this.states.length - 1; }

  push(state) {
    this.states.splice(this.at + 1, Infinity, state);
    if (this.states.length > this.limit) this.states.shift();
    this.at = this.states.length - 1;
  }

  // the state before (or after) the one shown, now the one shown; undefined if there is none
  undo() { return this.canUndo ? this.states[--this.at] : undefined; }

  redo() { return this.canRedo ? this.states[++this.at] : undefined; }
}

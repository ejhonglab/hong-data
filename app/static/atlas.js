// The AL atlas tab: the glomeruli of the Grabe et al. 2016 in vivo antennal lobe atlas (static/atlas/grabe2016.*,
// made by make_grabe2016.py beside them) as 3D meshes, drawn with three.js (vendored), to turn, zoom and pan; each
// glomerulus hidden or focused (the others not hidden then drawn nearly transparent), and arrows of the anatomical
// axes (from the glomeruli's names) that turn with the view.

const DATA = '/static/atlas/grabe2016';
const THREE_SRC = '/static/vendor/three/three.module.min.js';
const FADED = 0.07; // the opacity of glomeruli not focused, while some are
const ROTATE = 0.008; // radians per pixel dragged
// the axes' arrows: [the atlas' axis, its positive end's letter, its negative end's], each in a colour of AXIS_COLOURS
const AXES = { both: [['dorsal', 'D', 'V'], ['anterior', 'A', 'P'], ['left', 'L', 'R']],
  one: [['dorsal', 'D', 'V'], ['anterior', 'A', 'P'], ['medial', 'M', 'L']] };
const AXIS_COLOURS = ['#4e79a7', '#b07aa1', '#59a14f'];
// the mirroring each axis has (`flip`'s keys): with both lobes, 'left' (the fly's) is M/L's
const FLIP_OF = { dorsal: 'dv', anterior: 'ap', medial: 'ml', left: 'ml' };
// the views to turn to: from which side it is seen, and which way is up
const VIEWS = [['Anterior', 'anterior', 'dorsal'], ['Posterior', '-anterior', 'dorsal'], ['Dorsal', 'dorsal', '-anterior'], ['Ventral', '-dorsal', '-anterior'],
  ['Lateral', '-medial', 'dorsal'], ['Medial', 'medial', 'dorsal']];

export class AtlasPage {
  // helpers: { $, esc }
  constructor(h) {
    this.h = h;
    this.ready = null;
    this.side = 'right'; // the lobes shown: 'left', 'right' or 'both'
    this.flip = { dv: false, ap: false, ml: false }; // the axes the lobes shown are mirrored along (`_setFlips`)
    this.hidden = new Set(); // glomeruli (by name) hidden
    this.focused = new Set(); // and focused: the others drawn nearly transparent
    this.query = '';
    this.under = null; // the glomerulus under the pointer, which Space shows or hides: { name, from: 'view' | 'list', at (the pointer, of the view) }
  }

  // when the tab is shown: made the first time (three.js and the atlas read), else drawn at its size now
  load() {
    if (!this.ready) this.ready = this._init().catch((e) => { this.h.$('#atlas-status').textContent = `Could not show the atlas: ${e.message}`; });
    else this._resize();
    return this.ready;
  }

  async _init() {
    const { $ } = this.h;
    $('#atlas-status').textContent = 'Reading the atlas…';
    const get = async (url, how) => { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r[how](); };
    const [THREE, doc, bin] = await Promise.all([import(THREE_SRC), get(`${DATA}.json`, 'json'), get(`${DATA}.bin`, 'arrayBuffer')]);
    this.T = THREE;
    this.doc = doc;
    const box = $('#atlas-view');
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    box.prepend(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 1, 1, 5000);
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.1));
    const head = new THREE.DirectionalLight(0xffffff, 2.2); // (a headlight: lights whatever side is seen)
    head.position.set(0.3, 0.6, 1);
    this.camera.add(head);
    this.scene.add(this.camera);
    this.target = new THREE.Vector3();
    this.root = new THREE.Group(); // (the glomeruli, mirrored as `flip` says)
    this.root.matrixAutoUpdate = false;
    this.scene.add(this.root);
    this.meshes = doc.glomeruli.map((g) => {
      const geo = new THREE.BufferGeometry();
      const q = new Int16Array(bin, g.vertices[0], g.vertices[1] * 3), pos = new Float32Array(q.length);
      for (let i = 0; i < q.length; i++) pos[i] = q[i] / doc.quant;
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setIndex(new THREE.BufferAttribute(new Uint16Array(bin, g.triangles[0], g.triangles[1] * 3), 1));
      geo.computeVertexNormals();
      geo.computeBoundingSphere();
      const mesh = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({ color: g.colour, shininess: 25, specular: 0x222222, transparent: true }));
      mesh.userData = g;
      this.root.add(mesh);
      return mesh;
    });
    this.raycaster = new THREE.Raycaster();
    $('#atlas-views').innerHTML = VIEWS.map(([label, from, up]) => `<button type="button" class="btn small" data-view="${from}" data-up="${up}">${label}</button>`).join('');
    this._wire();
    new ResizeObserver(() => this._resize()).observe(box);
    $('#atlas-status').textContent = '';
    $('#atlas-source').innerHTML = `${this.h.esc(doc.citation)} Its labels as <a href="${this.h.esc(doc.source.split(' ')[0])}" target="_blank"
      rel="noopener">al_atlas</a> exports them, each glomerulus a smoothed surface of its voxels (${doc.smooth_um} µm). Colours: glomeruli within
      ${doc.near_um} µm of each other differ (${doc.neighbours_sharing_a_colour ? `but for ${doc.neighbours_sharing_a_colour} pairs` : 'all of them'}),
      the same glomerulus alike in both lobes. The axes are from the glomeruli's names (D, V; A, P, M, L as the second letter of D and V
      glomeruli), made perpendicular.`;
    this._setSide(this.side);
  }

  // ------------------------------------------------------------ controls
  _wire() {
    const { $ } = this.h, canvas = this.renderer.domElement;
    $('#atlas-side').addEventListener('change', (e) => this._setSide(e.target.value));
    $('#atlas-flips').addEventListener('change', (e) => {
      const c = e.target.closest('[data-flip]');
      if (c) this._setFlips({ ...this.flip, [c.dataset.flip]: c.checked });
    });
    $('#atlas-search').addEventListener('input', (e) => { this.query = e.target.value.trim().toLowerCase(); this._renderList(); });
    $('#atlas-show-all').addEventListener('click', () => { this.hidden.clear(); this._update(); });
    $('#atlas-hide-all').addEventListener('click', () => { for (const n of this._names()) this.hidden.add(n); this._update(); });
    $('#atlas-unfocus').addEventListener('click', () => { this.focused.clear(); this._update(); });
    $('#atlas-views').addEventListener('click', (e) => { const b = e.target.closest('[data-view]'); if (b) this._view(b.dataset.view, b.dataset.up); });
    $('#atlas-list').addEventListener('click', (e) => {
      const b = e.target.closest('[data-act]');
      if (!b) return;
      const name = b.closest('[data-name]').dataset.name, set = b.dataset.act === 'hide' ? this.hidden : this.focused;
      if (set.has(name)) set.delete(name); else set.add(name);
      if (b.dataset.act === 'focus') this.hidden.delete(name); // (one focused is shown)
      this._update();
    });
    // dragging turns it about its middle (with Shift, or the right button: moves it), the wheel zooms; a click focuses a glomerulus
    let drag = null;
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('pointerdown', (e) => {
      drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2, moved: 0 };
      try { canvas.setPointerCapture(e.pointerId); } catch { /* (a pointer no longer down) */ }
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!drag) { this._hover(e); return; }
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX;
      drag.y = e.clientY;
      drag.moved += Math.abs(dx) + Math.abs(dy);
      if (drag.pan) this._pan(dx, dy); else this._turn(dx, dy);
    });
    canvas.addEventListener('pointerup', (e) => {
      if (drag && drag.moved < 4 && e.button === 0) {
        const g = this._pick(e);
        if (g) { if (this.focused.has(g.name)) this.focused.delete(g.name); else this.focused.add(g.name); this._update(); }
      }
      drag = null;
    });
    canvas.addEventListener('pointerleave', () => {
      $('#atlas-tip').hidden = true;
      if (this.under?.from === 'view') this.under = null;
    });
    // Space: the glomerulus under the pointer hidden (in the view: so the one behind it comes next), or shown or hidden (in the list)
    $('#atlas-list').addEventListener('mouseover', (e) => {
      const row = e.target.closest('[data-name]');
      this.under = row ? { name: row.dataset.name, from: 'list' } : null;
    });
    $('#atlas-list').addEventListener('mouseleave', () => { if (this.under?.from === 'list') this.under = null; });
    document.addEventListener('keydown', (e) => {
      if (e.key !== ' ' || e.ctrlKey || e.metaKey || e.altKey || $('#tab-al-atlas').hidden || !this.under
        || e.target.closest?.('input, textarea, select, [contenteditable]')) return;
      e.preventDefault();
      const { name, from, at } = this.under;
      if (this.hidden.has(name)) this.hidden.delete(name); else this.hidden.add(name);
      this._update();
      if (from === 'view' && at) this._hover(at); // (now the one behind it, if any)
    });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const off = this.camera.position.clone().sub(this.target), d = off.length() * Math.exp(e.deltaY * 0.0012);
      off.setLength(Math.max(10, Math.min(2000, d)));
      this.camera.position.copy(this.target).add(off);
      this._draw();
    }, { passive: false });
  }

  // turns the view about the middle: across the screen about the camera's up, up and down about its right
  _turn(dx, dy) {
    const T = this.T, cam = this.camera, off = cam.position.clone().sub(this.target);
    const right = new T.Vector3().crossVectors(cam.up, off).normalize();
    const q = new T.Quaternion().setFromAxisAngle(cam.up.clone().normalize(), -dx * ROTATE)
      .multiply(new T.Quaternion().setFromAxisAngle(right, -dy * ROTATE));
    off.applyQuaternion(q);
    cam.up.applyQuaternion(q);
    cam.position.copy(this.target).add(off);
    cam.lookAt(this.target);
    this._draw();
  }

  _pan(dx, dy) {
    const T = this.T, cam = this.camera, off = cam.position.clone().sub(this.target);
    const per = (2 * off.length() * Math.tan((cam.fov * Math.PI) / 360)) / this.renderer.domElement.clientHeight;
    // (the glomeruli follow the pointer: the camera and its target move the other way)
    const right = new T.Vector3().crossVectors(cam.up, off).normalize().multiplyScalar(-dx * per);
    const move = right.add(cam.up.clone().normalize().multiplyScalar(dy * per));
    this.target.add(move);
    cam.position.add(move);
    this._draw();
  }

  // an axis of the lobes shown as they are drawn (the atlas' unit vector, the other way if mirrored along it), with
  // "-name" the opposite way
  _axis(name) {
    const key = name.replace('-', ''), v = this._atlasAxis(key);
    return (name.startsWith('-') !== this.flip[FLIP_OF[key] || 'ml']) ? v.negate() : v;
  }

  // an axis of the lobes shown, as the atlas has it ('medial': with both lobes, the fly's left)
  _atlasAxis(key) {
    const a = this.doc.axes[this.side];
    return new this.T.Vector3(...(a[key] || a.left));
  }

  // The mirrorings along the axes that, seen from dorsal with posterior up, put medial on the left and lateral on the
  // right (of both lobes: none): of the right lobe, along A/P, as the atlas is of a fly as it is (lateral on the right
  // is the left lobe's)
  _defaultFlips(side) {
    if (side === 'both') return { dv: false, ap: false, ml: false };
    const a = this.doc.axes[side], A = new this.T.Vector3(...a.anterior), D = new this.T.Vector3(...a.dorsal);
    const right = A.clone().negate().cross(D); // (the screen's right, seen from D with -A up)
    return { dv: false, ap: right.dot(new this.T.Vector3(...a.medial)) > 0, ml: false };
  }

  // mirrors the glomeruli along the axes ticked, about the middle of those of the lobes shown; the arrows follow
  _setFlips(flip) {
    const T = this.T;
    this.flip = { ...flip };
    const mid = new T.Vector3(), on = this.meshes.filter((m) => this._onSide(m));
    for (const m of on) mid.add(new T.Vector3(...m.userData.centroid));
    mid.divideScalar(Math.max(1, on.length));
    const mat = new T.Matrix4();
    for (const [key, f] of Object.entries(FLIP_OF)) {
      if (!this.flip[f] || key === (this.side === 'both' ? 'medial' : 'left')) continue;
      const v = this._atlasAxis(key), [x, y, z] = [v.x, v.y, v.z];
      // (a reflection through the plane through the middle across the axis: I - 2 v v')
      mat.premultiply(new T.Matrix4().set(1 - 2 * x * x, -2 * x * y, -2 * x * z, 0, -2 * x * y, 1 - 2 * y * y, -2 * y * z, 0,
        -2 * x * z, -2 * y * z, 1 - 2 * z * z, 0, 0, 0, 0, 1));
    }
    this.root.matrix.copy(new T.Matrix4().makeTranslation(mid.x, mid.y, mid.z).multiply(mat).multiply(new T.Matrix4().makeTranslation(-mid.x, -mid.y, -mid.z)));
    this.root.matrixWorldNeedsUpdate = true;
    this.root.updateMatrixWorld(true);
    for (const c of this.h.$('#atlas-flips').querySelectorAll('[data-flip]')) c.checked = this.flip[c.dataset.flip];
    this._draw();
  }

  // looks at the glomeruli shown from one side (an axis), with another up, at a distance they all fit in
  _view(from, up) {
    if (this.side === 'both' && /medial/.test(from)) from = from.startsWith('-') ? 'left' : '-left';
    const shown = this.meshes.filter((m) => m.visible), box = new this.T.Box3();
    for (const m of shown) box.expandByObject(m);
    const sphere = box.isEmpty() ? new this.T.Sphere(new this.T.Vector3(), 50) : box.getBoundingSphere(new this.T.Sphere());
    this.target.copy(sphere.center);
    // (the box's sphere is wider than the glomeruli are, seen from any side)
    const d = (sphere.radius * 0.8) / Math.sin((this.camera.fov * Math.PI) / 360);
    this.camera.position.copy(sphere.center).add(this._axis(from).multiplyScalar(d));
    this.camera.up.copy(this._axis(up));
    this.camera.lookAt(this.target);
    this._draw();
  }

  _setSide(side) {
    this.side = side;
    const both = side === 'both';
    for (const b of this.h.$('#atlas-views').querySelectorAll('[data-view]')) {
      const label = { '-medial': both ? "Fly's left" : 'Lateral', medial: both ? "Fly's right" : 'Medial' }[b.dataset.view];
      if (label) b.textContent = label;
    }
    this.h.$('#atlas-flip-ml-label').textContent = both ? 'L/R' : 'M/L';
    this._update(false);
    this._setFlips(this._defaultFlips(side));
    this._view('dorsal', '-anterior');
  }

  // ------------------------------------------------------------ drawing
  _names() { return [...new Set(this.meshes.filter((m) => this._onSide(m)).map((m) => m.userData.name))]; }

  _onSide(m) { return this.side === 'both' || m.userData.side === this.side; }

  // each glomerulus shown or not, and opaque or faded, as hidden and focused say; then the list, and the view
  _update(draw = true) {
    const some = this.focused.size > 0;
    for (const m of this.meshes) {
      const n = m.userData.name, faded = some && !this.focused.has(n);
      m.visible = this._onSide(m) && !this.hidden.has(n);
      m.material.opacity = faded ? FADED : 1;
      m.material.depthWrite = !faded;
      m.renderOrder = faded ? 1 : 0;
    }
    this._renderList();
    if (draw) this._draw();
  }

  _renderList() {
    const { $, esc } = this.h, q = this.query;
    const byName = new Map();
    for (const m of this.meshes) if (this._onSide(m) && !byName.has(m.userData.name)) byName.set(m.userData.name, m.userData);
    const names = [...byName.keys()].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
    const shown = names.filter((n) => !q || n.toLowerCase().includes(q) || (byName.get(n).receptor || '').toLowerCase().includes(q));
    $('#atlas-count').textContent = `${names.length} glomeruli${this.side === 'both' ? ' (each in both lobes)' : ''}: ${
      names.length - names.filter((n) => this.hidden.has(n)).length} shown${this.focused.size ? `, ${this.focused.size} focused` : ''}`;
    $('#atlas-list').innerHTML = shown.map((n) => {
      const g = byName.get(n), hid = this.hidden.has(n), foc = this.focused.has(n);
      return `<div class="atlas-row${hid ? ' hidden-row' : ''}${foc ? ' focused' : ''}" data-name="${esc(n)}">
        <span class="swatch-roi" style="background:${esc(g.colour)}"></span><span class="atlas-name">${esc(n)}</span>
        <span class="muted small">${esc(g.receptor || '')}</span>
        <button type="button" class="btn small" data-act="focus" aria-pressed="${foc}" title="Focus: the other glomeruli shown drawn nearly transparent (click it on the image too)">${foc ? 'Unfocus' : 'Focus'}</button>
        <button type="button" class="btn small" data-act="hide" aria-pressed="${hid}">${hid ? 'Show' : 'Hide'}</button></div>`;
    }).join('') || '<p class="muted small">None matches.</p>';
  }

  _resize() {
    if (!this.renderer) return;
    const box = this.h.$('#atlas-view'), w = box.clientWidth, h = box.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this._draw();
  }

  // draws once, at the next frame
  _draw() {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = null;
      this.renderer.render(this.scene, this.camera);
      this._drawAxes();
    });
  }

  // the axes' arrows, in the corner: each axis as the camera sees it (its ends' letters; those towards the viewer drawn over,
  // those away from it paler), so that they turn with the glomeruli
  _drawAxes() {
    const c = this.h.$('#atlas-axes'), dpr = window.devicePixelRatio || 1, size = c.clientWidth, high = c.clientHeight;
    if (c.width !== size * dpr) { c.width = size * dpr; c.height = high * dpr; }
    // (the arrows in a square at the bottom; above them, a line for an axis seen end on)
    const ctx = c.getContext('2d'), mid = size / 2, midY = high - size / 2, len = size * 0.34;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, high);
    const inv = this.camera.quaternion.clone().invert(), ink = getComputedStyle(document.body).getPropertyValue('--ink').trim() || '#222';
    const ends = [];
    (this.side === 'both' ? AXES.both : AXES.one).forEach(([key, pos, neg], i) => {
      const v = this._axis(key).applyQuaternion(inv);
      ends.push({ v, label: pos, colour: AXIS_COLOURS[i], head: true }, { v: v.clone().negate(), label: neg, colour: AXIS_COLOURS[i], head: false });
    });
    ends.sort((a, b) => a.v.z - b.v.z); // (farthest first)
    for (const e of ends) {
      const x = mid + e.v.x * len, y = midY - e.v.y * len, front = e.v.z >= 0;
      ctx.globalAlpha = 0.45 + 0.55 * (e.v.z + 1) / 2;
      ctx.strokeStyle = ctx.fillStyle = e.colour;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(mid, midY);
      ctx.lineTo(x, y);
      ctx.stroke();
      // an arrowhead at each end, pointing out
      const ang = Math.atan2(y - midY, x - mid), flat = Math.hypot(x - mid, y - midY) > 6;
      if (flat) {
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x - 8 * Math.cos(ang - 0.45), y - 8 * Math.sin(ang - 0.45));
        ctx.lineTo(x - 8 * Math.cos(ang + 0.45), y - 8 * Math.sin(ang + 0.45));
        ctx.closePath();
        ctx.fill();
      }
      ctx.font = `${front ? 700 : 500} ${front ? 14 : 12}px system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = front ? e.colour : ink;
      const lx = mid + e.v.x * (len + 13), ly = midY - e.v.y * (len + 13);
      ctx.fillText(e.label, lx, ly);
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = ink;
    ctx.beginPath();
    ctx.arc(mid, midY, 2.5, 0, 2 * Math.PI);
    ctx.fill();
    // (an axis seen end on: which end is towards the viewer, said above the arrows)
    const end = ends.filter((e) => e.v.z > 0.85).map((e) => e.label)[0];
    ctx.font = '600 12px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(end ? `${end} towards you` : '', 4, 9);
  }

  // the glomerulus under the pointer (of those shown; one focused before those faded), or null
  _pick(e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.raycaster.setFromCamera(new this.T.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), this.camera);
    const hits = this.raycaster.intersectObjects(this.meshes.filter((m) => m.visible), false);
    const opaque = hits.find((h) => h.object.material.opacity === 1);
    return (opaque || hits[0])?.object.userData || null;
  }

  _hover(e) {
    const { $, esc } = this.h, tip = $('#atlas-tip'), g = this._pick(e);
    this.under = g ? { name: g.name, from: 'view', at: { clientX: e.clientX, clientY: e.clientY } } : null;
    if (!g) { tip.hidden = true; return; }
    const r = $('#atlas-view').getBoundingClientRect();
    tip.innerHTML = `<b>${esc(g.name)}</b> <span class="muted">${esc(g.side)} lobe${g.receptor ? ` · ${esc(g.receptor)}` : ''} · ${Math.round(g.volume_um3)} µm³</span>`;
    tip.style.left = `${e.clientX - r.left + 14}px`;
    tip.style.top = `${e.clientY - r.top + 10}px`;
    tip.hidden = false;
  }
}

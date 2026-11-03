// Labels with LaTeX math in them, written as matplotlib's are: text, with math between $
// signs (or \( and \)), e.g. '$\frac{\Delta F}{F_0}$ (z-scored)', \$ for a dollar sign; or,
// with no $ but a TeX command in it, all math ('\Delta F/F'). The math is typeset by MathJax
// (vendored, loaded the first time a label has any), whose SVG is read into paths
// (`texGeometry`); a label is drawn on the page (`renderLabel`) and in exported figures
// (figure.js) from those, so nothing MathJax makes of a label goes in the page as it is.

const MATHJAX_SRC = '/static/vendor/mathjax/tex-svg-full.js';
// the x-height of MathJax's TeX fonts (in em), by which math is scaled to the text around it
// (as MathJax does): to that text's x-height
export const MATH_EX = 0.442;
const SVGNS = 'http://www.w3.org/2000/svg';

// A label as its parts: [{text} | {tex}], in order.
export function labelParts(s) {
  s = String(s ?? '');
  const parts = [];
  let text = '';
  const flush = () => { if (text) parts.push({ text }); text = ''; };
  for (let i = 0; i < s.length;) {
    if (s.startsWith('\\$', i)) { text += '$'; i += 2; continue; }
    const delim = s.startsWith('$$', i) ? ['$$', '$$'] : s[i] === '$' ? ['$', '$'] : s.startsWith('\\(', i) ? ['\\(', '\\)'] : null;
    const end = delim ? closing(s, i + delim[0].length, delim[1]) : -1;
    if (delim && end > i + delim[0].length) {
      flush();
      parts.push({ tex: s.slice(i + delim[0].length, end) });
      i = end + delim[1].length;
    } else {
      text += s[i++];
    }
  }
  flush();
  if (!parts.some((p) => p.tex != null) && /\\[A-Za-z]/.test(s)) return [{ tex: s }];
  return parts;
}

// where math opened before `from` closes (`close`, not escaped by a backslash), else -1
function closing(s, from, close) {
  for (let j = from; j < s.length; j += s[j] === '\\' ? 2 : 1) if (s.startsWith(close, j)) return j;
  return -1;
}

// A label as plain text, its math as written ('$\Delta F/F$').
export const labelText = (parts) => parts.map((p) => (p.tex != null ? `$${p.tex}$` : p.text)).join('');

let loading = null, ready = false;

// MathJax, loaded once (resolves when it can typeset).
function mathJax() {
  if (ready) return Promise.resolve(window.MathJax);
  if (!loading) {
    loading = new Promise((resolve, reject) => {
      window.MathJax = {
        startup: {
          typeset: false,
          ready() { window.MathJax.startup.defaultReady(); window.MathJax.startup.promise.then(() => { ready = true; resolve(window.MathJax); }); },
        },
        options: { enableMenu: false },
        // (no loading more of it, nor links, styles or classes in a label; errors thrown, not drawn)
        tex: { packages: { '[-]': ['require', 'autoload', 'html', 'action', 'noerrors', 'noundefined'] }, formatError: (jax, err) => { throw err; } },
        svg: { fontCache: 'none' },
      };
      const script = document.createElement('script');
      script.src = MATHJAX_SRC;
      script.async = true;
      script.onerror = () => { loading = null; reject(new Error('MathJax could not be loaded')); };
      document.head.appendChild(script);
    });
  }
  return loading;
}

const geometries = new Map();

// TeX math as drawn by MathJax (once loaded): {w, a, d: its width, and height above and depth
// below the baseline, in em; items}, or {error}. Items are in em, y down from the baseline:
// {path (SVG path data), m, fill (whether filled), stroke (its width, 0 for none)},
// {text, m, size, italic, bold} (a character MathJax's fonts lack), or {clip: [x, y, w, h],
// m, items} (only so much of its items shown), each m ([a, b, c, d, e, f], as SVG's) taking
// its coordinates to the label's.
export function texGeometry(tex) {
  let g = geometries.get(tex);
  if (!g) {
    try {
      g = readSvg(window.MathJax.tex2svg(tex, { display: false }).querySelector('svg'));
    } catch (e) {
      g = { error: e.message || String(e) };
    }
    if (geometries.size > 200) geometries.clear();
    geometries.set(tex, g);
  }
  return g;
}

// a label's parts with the geometry of their math (`g`), MathJax loaded first if need be
// (if it cannot be, each `g` is its error)
export async function labelGeometry(parts) {
  if (!parts.some((p) => p.tex != null)) return parts;
  try { await mathJax(); } catch (e) { return parts.map((p) => (p.tex != null ? { ...p, g: { error: e.message } } : p)); }
  return parts.map((p) => (p.tex != null ? { ...p, g: texGeometry(p.tex) } : p));
}

// A label shown in an element: its text as text, its math as SVG drawn from its geometry (as
// large as the text's x-height says, in its color). Resolves to what MathJax could not read
// in it (then shown as written, marked).
export async function renderLabel(el, label) {
  const parts = Array.isArray(label) ? label : labelParts(label);
  const seq = (el._labelSeq = (el._labelSeq || 0) + 1);
  if (!ready && parts.some((p) => p.tex != null)) el.textContent = labelText(parts); // (while MathJax loads)
  const shown = await labelGeometry(parts);
  if (seq !== el._labelSeq) return [];
  const errors = [];
  el.replaceChildren(...shown.map((p) => {
    if (p.tex == null) return document.createTextNode(p.text);
    if (!p.g.error) return mathElement(p.g);
    errors.push(p.g.error);
    const span = document.createElement('span');
    span.className = 'tex-error';
    span.title = p.g.error;
    span.textContent = `$${p.tex}$`;
    return span;
  }));
  return errors;
}

function mathElement(g) {
  const svg = document.createElementNS(SVGNS, 'svg'), ex = (v) => `${+(v / MATH_EX).toFixed(3)}ex`;
  svg.setAttribute('class', 'tex');
  svg.setAttribute('width', ex(g.w));
  svg.setAttribute('height', ex(g.a + g.d));
  svg.setAttribute('viewBox', `${fmt(0)} ${fmt(-g.a)} ${fmt(g.w)} ${fmt(g.a + g.d)}`);
  svg.setAttribute('aria-hidden', 'true');
  svg.style.verticalAlign = ex(-g.d);
  svg.innerHTML = geometrySvg(g.items);
  return svg;
}

let clipIds = 0;
const xmlEsc = (s) => String(s).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
// a number as written in SVG or PDF: never with an exponent
export const fmt = (v) => String(+v.toFixed(6));

// SVG markup of a geometry's items (in its coordinates), in a color
export function geometrySvg(items, color = 'currentColor') {
  return items.map((it) => {
    const m = `transform="matrix(${it.m.map(fmt).join(' ')})"`;
    if (it.clip) {
      const id = `tex-clip-${++clipIds}`, [x, y, w, h] = it.clip.map(fmt);
      return `<clipPath id="${id}"><rect ${m} x="${x}" y="${y}" width="${w}" height="${h}"/></clipPath>`
        + `<g clip-path="url(#${id})">${geometrySvg(it.items, color)}</g>`;
    }
    if (it.text != null) {
      return `<text ${m} font-size="${fmt(it.size)}" font-family="serif"${it.italic ? ' font-style="italic"' : ''}${it.bold ? ' font-weight="bold"' : ''}`
        + ` fill="${color}">${xmlEsc(it.text)}</text>`;
    }
    const paint = `fill="${it.fill ? color : 'none'}"${it.stroke ? ` stroke="${color}" stroke-width="${fmt(it.stroke)}"` : ''}`;
    return `<path ${m} d="${xmlEsc(it.path)}" ${paint}/>`;
  }).join('');
}

// ------------------------------------------------------------- reading MathJax's SVG
// [a, b, c, d, e, f] of SVG's transforms: A then B (as A·B)
export const mul = (A, B) => [A[0] * B[0] + A[2] * B[1], A[1] * B[0] + A[3] * B[1], A[0] * B[2] + A[2] * B[3],
  A[1] * B[2] + A[3] * B[3], A[0] * B[4] + A[2] * B[5] + A[4], A[1] * B[4] + A[3] * B[5] + A[5]];

export function invert([a, b, c, d, e, f]) {
  const det = a * d - b * c;
  return [d / det, -b / det, -c / det, a / det, (c * f - d * e) / det, (b * e - a * f) / det];
}

function transformOf(s) {
  let m = [1, 0, 0, 1, 0, 0];
  for (const [, fn, args] of (s || '').matchAll(/(\w+)\s*\(([^)]*)\)/g)) {
    const a = args.split(/[\s,]+/).filter(Boolean).map(Number);
    if (fn === 'translate') m = mul(m, [1, 0, 0, 1, a[0] || 0, a[1] || 0]);
    else if (fn === 'scale') m = mul(m, [a[0], 0, 0, a[1] ?? a[0], 0, 0]);
    else if (fn === 'matrix' && a.length === 6) m = mul(m, a);
    else if (fn === 'rotate') {
      const r = (a[0] * Math.PI) / 180, [cx, cy] = [a[1] || 0, a[2] || 0];
      m = mul(mul(mul(m, [1, 0, 0, 1, cx, cy]), [Math.cos(r), Math.sin(r), -Math.sin(r), Math.cos(r), 0, 0]), [1, 0, 0, 1, -cx, -cy]);
    }
  }
  return m;
}

// MathJax's SVG: 1000 units to the em, its viewBox from its top (-height above the baseline)
function readSvg(svg) {
  const [x0, y0, w, h] = svg.getAttribute('viewBox').split(/[\s,]+/).map(Number);
  const items = [];
  walk(svg.children, [0.001, 0, 0, 0.001, -x0 / 1000, 0], { fill: true, stroke: true, width: 0 }, items);
  return { w: w / 1000, a: -y0 / 1000, d: (h + y0) / 1000, items };
}

// the shapes of elements (and of their children), with their transforms (`m`, so far) and
// paint (inherited: whether filled and stroked, and the stroke's width), into `out`
function walk(nodes, m, paint, out) {
  for (const el of nodes) {
    const at = (k, dflt = 0) => { const v = parseFloat(el.getAttribute(k)); return Number.isFinite(v) ? v : dflt; };
    const own = mul(m, transformOf(el.getAttribute('transform')));
    const fill = el.getAttribute('fill'), stroke = el.getAttribute('stroke');
    const p = { fill: fill == null ? paint.fill : fill !== 'none', stroke: stroke == null ? paint.stroke : stroke !== 'none',
      width: el.hasAttribute('stroke-width') ? at('stroke-width') : paint.width };
    const shape = (path) => {
      if (path.trim() && (p.fill || (p.stroke && p.width > 0))) out.push({ path, m: own, fill: p.fill, stroke: p.stroke ? p.width : 0 });
    };
    switch (el.localName) {
      case 'g': walk(el.children, own, p, out); break;
      case 'svg': { // a window onto its contents, which shows only so much of them
        const [x, y, w, h] = ['x', 'y', 'width', 'height'].map((k) => at(k));
        const vb = (el.getAttribute('viewBox') || `0 0 ${w} ${h}`).split(/[\s,]+/).map(Number);
        const sx = vb[2] ? w / vb[2] : 1, sy = vb[3] ? h / vb[3] : 1, items = [];
        walk(el.children, mul(own, [sx, 0, 0, sy, x - vb[0] * sx, y - vb[1] * sy]), p, items);
        if (items.length) out.push({ clip: [x, y, w, h], m: own, items });
        break;
      }
      case 'path': shape(el.getAttribute('d') || ''); break;
      case 'rect': // (but the background of \colorbox, its own color)
        if (fill && fill !== 'none' && fill !== 'currentColor') break;
        shape(`M${at('x')} ${at('y')}h${at('width')}v${at('height')}h${-at('width')}Z`);
        break;
      case 'line': p.fill = false; shape(`M${at('x1')} ${at('y1')}L${at('x2')} ${at('y2')}`); break;
      case 'polygon': case 'polyline': {
        const pts = (el.getAttribute('points') || '').trim().split(/[\s,]+/).map(Number);
        if (pts.length >= 4) shape(`M${pts.join(' ')}${el.localName === 'polygon' ? 'Z' : ''}`);
        break;
      }
      case 'ellipse': case 'circle': {
        const cx = at('cx'), cy = at('cy'), rx = el.localName === 'circle' ? at('r') : at('rx'), ry = el.localName === 'circle' ? rx : at('ry');
        const k = 0.5523; // (four cubic Béziers)
        shape(`M${cx + rx} ${cy}C${cx + rx} ${cy + k * ry} ${cx + k * rx} ${cy + ry} ${cx} ${cy + ry}C${cx - k * rx} ${cy + ry} ${cx - rx} ${cy + k * ry} ${cx - rx} ${cy}`
          + `C${cx - rx} ${cy - k * ry} ${cx - k * rx} ${cy - ry} ${cx} ${cy - ry}C${cx + k * rx} ${cy - ry} ${cx + rx} ${cy - k * ry} ${cx + rx} ${cy}Z`);
        break;
      }
      case 'text':
        if (el.textContent) {
          out.push({ text: el.textContent, m: mul(own, [1, 0, 0, 1, at('x'), at('y')]), size: at('font-size', 1000),
            italic: el.getAttribute('font-style') === 'italic', bold: el.getAttribute('font-weight') === 'bold' });
        }
        break;
      default: // (title, defs, ...)
    }
  }
}

// SVG path data as PDF's path operators (in the same coordinates; quadratic curves as cubic)
export function pdfPath(d) {
  const toks = d.match(/[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?/g) || [];
  const ops = [], P = (x, y) => `${fmt(x)} ${fmt(y)}`;
  let i = 0, cmd = '', x = 0, y = 0, x0 = 0, y0 = 0, prevQ = null, prevC = null;
  const n = () => +toks[i++];
  const quad = (qx, qy, ex, ey) => {
    ops.push(`${P(x + (2 / 3) * (qx - x), y + (2 / 3) * (qy - y))} ${P(ex + (2 / 3) * (qx - ex), ey + (2 / 3) * (qy - ey))} ${P(ex, ey)} c`);
    [x, y] = [ex, ey];
  };
  while (i < toks.length) {
    if (/[A-Za-z]/.test(toks[i])) cmd = toks[i++];
    const rel = cmd !== cmd.toUpperCase(), ox = rel ? x : 0, oy = rel ? y : 0;
    let q = null, c = null;
    switch (cmd.toUpperCase()) {
      case 'M': x = ox + n(); y = oy + n(); [x0, y0] = [x, y]; ops.push(`${P(x, y)} m`); cmd = rel ? 'l' : 'L'; break;
      case 'L': x = ox + n(); y = oy + n(); ops.push(`${P(x, y)} l`); break;
      case 'H': x = ox + n(); ops.push(`${P(x, y)} l`); break;
      case 'V': y = oy + n(); ops.push(`${P(x, y)} l`); break;
      case 'C': case 'S': {
        const [x1, y1] = cmd.toUpperCase() === 'C' ? [ox + n(), oy + n()] : prevC ? [2 * x - prevC[0], 2 * y - prevC[1]] : [x, y];
        const x2 = ox + n(), y2 = oy + n();
        x = ox + n(); y = oy + n();
        ops.push(`${P(x1, y1)} ${P(x2, y2)} ${P(x, y)} c`);
        c = [x2, y2];
        break;
      }
      case 'Q': case 'T': {
        q = cmd.toUpperCase() === 'Q' ? [ox + n(), oy + n()] : prevQ ? [2 * x - prevQ[0], 2 * y - prevQ[1]] : [x, y];
        quad(q[0], q[1], ox + n(), oy + n());
        break;
      }
      case 'A': i += 5; x = ox + n(); y = oy + n(); ops.push(`${P(x, y)} l`); break; // (as a line: MathJax draws none)
      case 'Z': ops.push('h'); [x, y] = [x0, y0]; cmd = ''; break;
      default: i++;
    }
    prevQ = q; prevC = c;
  }
  return ops.join(' ');
}

// Explanations of terms, shown in a box on hover (or focus) and kept open by a click until closed
// (its ×, Escape, or a click elsewhere); and lists that a link opens in a box to search.

const escAttr = (s) => String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');

// a term (HTML) explained by `text` (plain text; blank lines split it into paragraphs)
export const tip = (label, text) => `<span class="tip" tabindex="0" role="button" data-tip="${escAttr(text)}">${label}</span>`;

// a link (its HTML) that opens `items` in a box to search, under `title`
export const listLink = (label, title, items) => `<a href="#" class="tip-list" data-title="${escAttr(title)}"
  data-list="${escAttr(JSON.stringify(items))}">${label}</a>`;

let box = null, anchor = null, pinned = false;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function hide() {
  if (box) box.hidden = true;
  anchor = null;
  pinned = false;
}

// shows `content` (nodes) in the box under `at`, with a close button if pinned
function show(at, content, pin) {
  if (!box) {
    box = el('div', 'tip-box');
    box.setAttribute('role', 'tooltip');
    document.body.append(box);
  }
  anchor = at;
  pinned = pin;
  const close = el('button', 'tip-close', '×');
  close.type = 'button';
  close.title = 'Close (Esc)';
  close.addEventListener('click', hide);
  box.replaceChildren(...(pin ? [close] : []), ...content);
  box.classList.toggle('pinned', pin);
  box.hidden = false;
  const r = at.getBoundingClientRect(), w = box.offsetWidth;
  const left = Math.max(8, Math.min(r.left, document.documentElement.clientWidth - w - 8));
  const below = r.bottom + 6 + box.offsetHeight <= window.innerHeight || r.top < box.offsetHeight + 6;
  box.style.left = `${left + window.scrollX}px`;
  box.style.top = `${(below ? r.bottom + 6 : r.top - 6 - box.offsetHeight) + window.scrollY}px`;
}

const paragraphs = (text) => text.split(/\n\s*\n/).map((t) => el('p', null, t.trim()));

function showList(at) {
  let items = [];
  try { items = JSON.parse(at.dataset.list); } catch { /* none */ }
  const search = el('input', 'tip-search');
  search.type = 'search';
  search.placeholder = 'Search';
  search.setAttribute('aria-label', 'Search the list');
  const ul = el('ul', 'tip-items'), count = el('p', 'tip-count muted');
  const render = () => {
    const q = search.value.trim().toLowerCase(), shown = items.filter((s) => String(s).toLowerCase().includes(q));
    ul.replaceChildren(...shown.map((s) => el('li', null, s)));
    count.textContent = q ? `${shown.length} of ${items.length}` : `${items.length}`;
  };
  search.addEventListener('input', render);
  render();
  show(at, [el('p', 'tip-title', at.dataset.title || ''), search, count, ul], true);
  search.focus();
}

document.addEventListener('mouseover', (e) => {
  const t = e.target.closest?.('.tip');
  if (t && !pinned && t !== anchor) show(t, paragraphs(t.dataset.tip), false);
});
document.addEventListener('mouseout', (e) => {
  const t = e.target.closest?.('.tip');
  if (t && !pinned && t === anchor && !t.contains(e.relatedTarget)) hide();
});
document.addEventListener('focusin', (e) => {
  const t = e.target.closest?.('.tip');
  if (t && !pinned) show(t, paragraphs(t.dataset.tip), false);
});
document.addEventListener('focusout', (e) => {
  if (!pinned && e.target === anchor) hide();
});
document.addEventListener('click', (e) => {
  const t = e.target.closest?.('.tip, .tip-list');
  if (t) {
    e.preventDefault();
    if (pinned && t === anchor) hide();
    else if (t.classList.contains('tip-list')) showList(t);
    else show(t, paragraphs(t.dataset.tip), true);
  } else if (pinned && box && !box.contains(e.target)) hide();
});
document.addEventListener('keydown', (e) => {
  const t = e.target.closest?.('.tip');
  if (t && (e.key === 'Enter' || e.key === ' ')) {
    e.preventDefault();
    if (pinned && t === anchor) hide(); else show(t, paragraphs(t.dataset.tip), true);
  } else if (e.key === 'Escape' && box && !box.hidden) {
    const back = pinned ? anchor : null;
    hide();
    back?.focus?.();
  }
});

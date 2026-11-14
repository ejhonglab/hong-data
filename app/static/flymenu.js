// A menu of flies to choose one from (AL segmentation tab), in place of a <select>: a box to search
// them over the list, and each fly with what is known of it in words, some coloured. Made in a
// <details class="menu fly-menu"> holding a <summary>, and in a .menu-list an <input type="search">
// and a .fly-menu-items (app.js closes an open .menu on a press outside it, or Escape).

export class FlyMenu {
  // onPick(value): called when a fly is chosen by hand
  constructor(root, { esc, onPick }) {
    Object.assign(this, { root, esc, onPick, groups: [], extra: [], value: null });
    this.summary = root.querySelector('summary');
    this.search = root.querySelector('input[type=search]');
    this.items = root.querySelector('.fly-menu-items');
    root.addEventListener('toggle', () => {
      if (!root.open) return;
      this.search.value = '';
      this._renderItems();
      this.search.focus();
      this.items.querySelector('.active')?.scrollIntoView({ block: 'nearest' });
    });
    this.search.addEventListener('input', () => this._renderItems());
    this.items.addEventListener('click', (e) => { const b = e.target.closest('[data-value]'); if (b) this._pick(b.dataset.value); });
    // Enter in the search box takes the first fly found; Up and Down go through those listed
    root.addEventListener('keydown', (e) => {
      const buttons = [...this.items.querySelectorAll('[data-value]')], at = buttons.indexOf(document.activeElement);
      if (e.key === 'Enter' && e.target === this.search) {
        e.preventDefault();
        const first = buttons.find((b) => !('extra' in b.dataset));
        if (first) this._pick(first.dataset.value);
      } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        const to = at + (e.key === 'ArrowDown' ? 1 : -1);
        (to < 0 ? this.search : buttons[Math.min(to, buttons.length - 1)])?.focus();
      }
    });
  }

  // What to choose from: `groups` [{ label, items: [{ value, label, search (what the search box is
  // matched in, beside its label), tags: [{ text, cls }] }] }], then `extra` [{ value, label }]: entries
  // listed whatever is searched for. `note`: said under them (e.g. that flies are still looked for).
  set(groups, extra = [], note = '') {
    Object.assign(this, { groups, extra, note });
    this._renderItems();
    this._renderSummary();
  }

  // every value that can be chosen
  values() { return [...this.groups.flatMap((g) => g.items), ...this.extra].map((it) => it.value); }

  // chooses a fly (not by hand: `onPick` is not called)
  choose(value) {
    this.value = value;
    this._renderItems();
    this._renderSummary();
  }

  _find(value) { return [...this.groups.flatMap((g) => g.items), ...this.extra].find((it) => it.value === value); }

  _html(it) {
    return `${this.esc(it.label)}${(it.tags || []).map((t) => ` <span class="fly-tag">· <span class="${t.cls || ''}">${this.esc(t.text)}</span></span>`).join('')}`;
  }

  _renderSummary() {
    const it = this._find(this.value);
    this.summary.innerHTML = `<span class="fly-menu-chosen">${it ? this._html(it) : '<span class="muted">Choose a fly</span>'}</span>`;
  }

  _renderItems() {
    const q = this.search.value.trim().toLowerCase();
    const button = (it, extra) => `<button type="button" data-value="${this.esc(it.value)}"${extra === true ? ' data-extra' : ''}${it.value === this.value ? ' class="active"' : ''}>${this._html(it)}</button>`;
    const groups = this.groups.map((g) => {
      const found = g.items.filter((it) => !q || `${it.search || ''} ${it.label}`.toLowerCase().includes(q));
      return found.length ? `<div class="fly-menu-group">${this.esc(g.label)}</div>${found.map((it) => button(it)).join('')}` : '';
    }).join('');
    this.items.innerHTML = `${groups || (q ? '<p class="menu-note">No fly matches.</p>' : '')}${this.note ? `<p class="menu-note">${this.esc(this.note)}</p>` : ''}
      ${this.extra.map((it) => button(it, true)).join('')}`;
  }

  _pick(value) {
    this.root.open = false;
    this.choose(value);
    this.summary.focus();
    this.onPick(value);
  }
}

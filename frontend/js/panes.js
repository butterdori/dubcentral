/* Collapsible panes: click the caret in a pane header to collapse it to just
   its title bar; the other pane in the same column takes the freed space
   (grid rules in style.css via :has()). State persists per pane. */
(() => {
  const KEY = 'dubconsole-collapsed';
  let state = {};
  try { state = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (_) {}
  document.querySelectorAll('.pane[data-pane]').forEach(pane => {
    const id = pane.dataset.pane;
    const btn = pane.querySelector('.pane-toggle');
    const h2 = pane.querySelector('h2');
    const apply = () => {
      const c = !!state[id];
      pane.classList.toggle('collapsed', c);
      btn.textContent = c ? '▸' : '▾';
      btn.setAttribute('aria-label', c ? 'Expand pane' : 'Collapse pane');
    };
    const toggle = () => {
      state[id] = !state[id];
      localStorage.setItem(KEY, JSON.stringify(state));
      apply();
    };
    btn.addEventListener('click', toggle);
    // double-click anywhere on the title bar also toggles
    h2.addEventListener('dblclick', toggle);
    apply();
  });
})();

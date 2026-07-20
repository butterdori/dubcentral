/* Theme switching. Loaded in <head> (before body paints) so the stored theme
   applies without a flash of the default. Persisted in localStorage — this is
   a self-hosted single-user app, so that's exactly the right scope. */
(() => {
  const THEMES = [
    ['nord-light', 'Nord Light'],
    ['nord-dark', 'Nord Dark'],
    ['catppuccin', 'Catppuccin'],
    ['dracula', 'Dracula'],
  ];
  const KEY = 'dubconsole-theme';
  const DEFAULT = 'nord-light';

  const valid = t => THEMES.some(([id]) => id === t);
  const stored = localStorage.getItem(KEY);
  const current = valid(stored) ? stored : DEFAULT;
  document.documentElement.dataset.theme = current;

  // populate any .theme-pick select once the DOM exists
  window.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('select.theme-pick').forEach(sel => {
      for (const [id, label] of THEMES) {
        const o = document.createElement('option');
        o.value = id; o.textContent = label;
        sel.append(o);
      }
      sel.value = current;
      sel.addEventListener('change', () => {
        document.documentElement.dataset.theme = sel.value;
        localStorage.setItem(KEY, sel.value);
      });
    });
  });
})();

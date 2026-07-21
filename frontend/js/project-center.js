/* Project Center: card grid (thumbnail, stats, dual-voice completion bar) */

async function refresh() {
  const { projects } = await api.get('/api/projects');
  const grid = $('#projects');
  grid.replaceChildren();
  $('#empty').hidden = projects.length > 0;
  $('#count').textContent = projects.length
    ? `${projects.length} project${projects.length > 1 ? 's' : ''}` : '';

  for (const p of projects) grid.append(card(p));
}

function card(p) {
  const s = p.stats;
  const open = () => { location.href = `/project.html?p=${encodeURIComponent(p.key)}`; };

  const thumb = el('div', { class: 'thumb', onclick: open, role: 'button',
                            tabindex: '0', title: 'Open project' });
  if (p.has_thumbnail) {
    thumb.style.backgroundImage =
      `url(/api/projects/${encodeURIComponent(p.key)}/thumbnail?ts=${p.modified_at})`;
  } else {
    thumb.textContent = p.has_video ? '…' : 'no video yet';
  }

  const done = s.n_dubbable ? (s.n_dubbed / s.n_dubbable) * 100 : 0;
  const bar = el('div', { class: 'voicebar',
    title: `${s.n_dubbed}/${s.n_dubbable} dubbable lines dubbed` },
    Object.assign(el('i'), { style: `width:${done}%` }));

  const stats = el('div', { class: 'stats' },
    `${fmtDur(s.duration_s)} · ${s.n_speakers} spk · ${s.n_lines} lines · ` +
    `${s.n_dubbed}/${s.n_dubbable} dubbed`);

  const when = el('span', { class: 'dim mono', style: 'font-size:11.5px' },
    new Date(p.modified_at * 1000).toLocaleString());
  const delBtn = el('button', { class: 'danger', onclick: () => confirmDelete(p) }, 'Delete');

  return el('div', { class: 'card' },
    thumb,
    el('div', { class: 'body' },
      el('div', { class: 'name', onclick: open }, p.name),
      bar, stats,
      el('div', { class: 'row' }, when, delBtn)));
}

function confirmDelete(p) {
  const dlg = $('#confirm-dlg');
  $('#confirm-title').textContent = `Delete "${p.name}"?`;
  $('#confirm-text').textContent =
    'Removes the project folder: video, subtitles, extracted clips, and every ' +
    'generated take. There is no undo for this.';
  $('#confirm-ok').onclick = async () => {
    dlg.close();
    try { await api.del(`/api/projects/${encodeURIComponent(p.key)}`); }
    catch (e) { alert(e.message); }
    refresh();
  };
  $('#confirm-cancel').onclick = () => dlg.close();
  dlg.showModal();
}

$('#create-form').addEventListener('submit', async ev => {
  ev.preventDefault();
  const name = $('#new-name').value.trim();
  if (!name) return;
  try {
    const meta = await api.postJSON('/api/projects', { name });
    location.href = `/project.html?p=${encodeURIComponent(meta.key)}`;
  } catch (e) { alert(e.message); }
});

refresh();

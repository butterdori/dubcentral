/* Upload Center: video player + uploads + CSV editor.
   Destructive actions (SRT upload onto existing lines, CSV editor save, CSV
   re-upload) confirm before firing — the API resets unconditionally. */

const KEY = new URLSearchParams(location.search).get('p');
if (!KEY) location.href = '/';
const P = `/api/projects/${encodeURIComponent(KEY)}`;

let meta = null;
let videoSrc = 'original';   // 'original' | 'dubbed' — sticky across refreshes
window.refreshPanes = () => {
  window.subCenter?.reload();
  window.speakerCenter?.reload();
  window.dubCenter?.reload();
};

function status(msg, kind = '') {
  const n = $('#uc-status');
  n.textContent = msg;
  n.className = 'uc-status ' + kind;
}

function confirmDlg(title, text) {
  return new Promise(resolve => {
    const dlg = $('#confirm-dlg');
    $('#confirm-title').textContent = title;
    $('#confirm-text').textContent = text;
    $('#confirm-ok').onclick = () => { dlg.close(); resolve(true); };
    $('#confirm-cancel').onclick = () => { dlg.close(); resolve(false); };
    dlg.showModal();
  });
}

async function refresh() {
  meta = await api.get(P);
  $('#crumb').textContent = meta.name;
  document.title = `Dub Console — ${meta.name}`;

  const player = $('#player');
  const toggle = $('#uc-src-toggle');
  const origBtn = $('#btn-src-original');
  const dubBtn = $('#btn-src-dubbed');
  if (meta.has_video) {
    toggle.hidden = false;
    dubBtn.disabled = !meta.has_final;
    dubBtn.title = meta.has_final ? 'Play the Final Remix output'
                                  : 'Run Final Remix first';
    if (videoSrc === 'dubbed' && !meta.has_final) videoSrc = 'original';  // fell out of date
    const src = videoSrc === 'dubbed'
      ? `${P}/remix/video?ts=${meta.modified_at}`
      : `${P}/video?ts=${meta.modified_at}`;
    if (!player.src.endsWith(src)) player.src = src;
    origBtn.classList.toggle('primary', videoSrc === 'original');
    dubBtn.classList.toggle('primary', videoSrc === 'dubbed');
    player.hidden = false;
    $('#no-video').hidden = true;
  } else {
    toggle.hidden = true;
    player.hidden = true;
    $('#no-video').hidden = false;
  }

  const s = meta.stats;
  const info = $('#seg-info');
  if (s.n_lines) {
    info.hidden = false;
    info.replaceChildren(
      el('span', {}, `segments: ${s.n_lines} lines · ${s.n_dubbable} dubbable · ${s.n_speakers} speakers`),
      el('span', { class: 'dim' }, `${s.n_dubbed}/${s.n_dubbable} dubbed`));
  } else {
    info.hidden = true;
  }
  const csvReady = s.n_lines > 0;
  $('#btn-edit-csv').disabled = !csvReady;
  $('#btn-dl-csv').disabled = !csvReady;
}

$('#btn-src-original').onclick = () => { videoSrc = 'original'; refresh(); };
$('#btn-src-dubbed').onclick = () => {
  if ($('#btn-src-dubbed').disabled) return;
  videoSrc = 'dubbed'; refresh();
};

/* live-enable "Dubbed" the moment a remix job finishes, without a manual
   reload — deferred to window 'load' since this script runs before
   speaker-center.js defines window.jobWatch */
window.addEventListener('load', () => {
  const refreshedFor = new Set();   // once per job id — the finished job
                                    // stays in /api/jobs/current forever
  let firstPoll = true;
  window.jobWatch?.on(s => {
    const j = s.job;
    if (firstPoll) {   // a remix completed before this page view: don't re-refresh
      firstPoll = false;
      if (j) refreshedFor.add(j.id);
    }
    if (!j || j.project_key !== KEY || j.kind !== 'remix') return;
    if (j.status !== 'done' || refreshedFor.has(j.id)) return;
    refreshedFor.add(j.id);
    refresh();
  });
});

/* ---- video ---- */
$('#btn-video').onclick = () => $('#file-video').click();
$('#file-video').onchange = async ev => {
  const f = ev.target.files[0];
  if (!f) return;
  status(`uploading ${f.name} (${(f.size / 1e6).toFixed(1)} MB)…`);
  try {
    await api.upload(`${P}/upload/video`, f);
    status('video uploaded ✓', 'ok');
    await refresh();
  } catch (e) { status(e.message, 'err'); }
  ev.target.value = '';
};

/* ---- SRT ---- */
$('#btn-srt').onclick = async () => {
  if (meta.stats.n_lines > 0) {
    const ok = await confirmDlg('Replace subtitles?',
      'Uploading a new SRT rebuilds the line table from scratch: speaker ' +
      'assignments, overrides, and all dub progress are reset; extracted ' +
      'clips and generated takes are removed.');
    if (!ok) return;
  }
  $('#file-srt').click();
};
$('#file-srt').onchange = async ev => {
  const f = ev.target.files[0];
  if (!f) return;
  status('running prep on ' + f.name + '…');
  try {
    const out = await api.upload(`${P}/upload/srt`, f);
    status(`prep done — ${out.n_lines} lines ✓`, 'ok');
    await refresh();
    window.refreshPanes();
  } catch (e) { status(e.message, 'err'); }
  ev.target.value = '';
};

/* ---- original-language SRT (additive, no reset) ---- */
$('#btn-orig-srt').onclick = () => $('#file-orig-srt').click();
$('#file-orig-srt').onchange = async ev => {
  const f = ev.target.files[0];
  if (!f) return;
  status('aligning original SRT…');
  try {
    const out = await api.upload(`${P}/upload/original_srt`, f);
    const bits = [`${out.matched_by_index} by index`];
    if (out.matched_by_timestamp) bits.push(`${out.matched_by_timestamp} by timestamp`);
    if (out.unmatched_source_entries) bits.push(`${out.unmatched_source_entries} unmatched`);
    status(`original text attached (${bits.join(', ')})` +
           (out.lines_without_original_text
             ? ` — ${out.lines_without_original_text} lines still without` : ' ✓'),
           out.lines_without_original_text ? '' : 'ok');
    window.refreshPanes();
  } catch (e) { status(e.message, 'err'); }
  ev.target.value = '';
};

/* ---- CSV editor ---- */
$('#btn-edit-csv').onclick = async () => {
  try {
    $('#csv-text').value = await api.get(`${P}/csv`);
    $('#csv-dlg').showModal();
  } catch (e) { status(e.message, 'err'); }
};
$('#csv-cancel').onclick = () => $('#csv-dlg').close();
$('#csv-save').onclick = async () => {
  try {
    const out = await api.putText(`${P}/csv`, $('#csv-text').value);
    $('#csv-dlg').close();
    status(`CSV saved — line table rebuilt (${out.n_lines} lines) ✓`, 'ok');
    await refresh();
    window.refreshPanes();
  } catch (e) {
    // keep the dialog open so the edit isn't lost; surface the row error
    status(e.message, 'err');
    alert(e.message);
  }
};

/* ---- CSV download / re-upload ---- */
$('#btn-dl-csv').onclick = () => { location.href = `${P}/csv`; };

$('#btn-ul-csv').onclick = async () => {
  const ok = await confirmDlg('Upload segments.csv?',
    'This wipes the current line table and starts fresh from the uploaded ' +
    'file: speakers, overrides, and all dub progress are reset; extracted ' +
    'clips and generated takes are removed.');
  if (ok) $('#file-csv').click();
};
$('#file-csv').onchange = async ev => {
  const f = ev.target.files[0];
  if (!f) return;
  try {
    const out = await api.upload(`${P}/upload/csv`, f);
    status(`CSV imported — ${out.n_lines} lines ✓`, 'ok');
    await refresh();
    window.refreshPanes();
  } catch (e) { status(e.message, 'err'); }
  ev.target.value = '';
};

window.uploadCenter = { refresh };
refresh().catch(e => {
  status(e.message, 'err');
  $('#crumb').textContent = 'project not found';
});

/* Sub Center: Diarization. Assignments are LOCAL until Save. Clip extraction and
   dubbing only sees saved state. Revert rebuilds from the original
   uploaded SRT; Download exports the current lines as an SRT.*/

(() => {
  const root = $('#subs-body');
  let rows = [];                 // grid rows from the server
  let speakers = {};             // key -> display name
  const pending = new Map();     // line_no -> speaker key|null (unsaved)
  const sel = new Set();

  const ADD = '__add__';

  function fmtTs(s) {
    const m = Math.floor(s / 60);
    const sec = (s - m * 60).toFixed(1).padStart(4, '0');
    return `${m}:${sec}`;
  }

  function isDirty() { return pending.size > 0; }

  function markDirty() {
    $('#subs-dirty').textContent = pending.size
      ? `● ${pending.size} unsaved` : '';
  }

  async function reload() {
    const g = await api.get(`${P}/dub/grid`);
    rows = g.rows;
    speakers = g.speakers;
    pending.clear();
    sel.clear();
    render();
  }

  /* ------------------------------ render ------------------------------ */

  function render() {
    markDirty();
    if (!rows.length) {
      root.replaceChildren(el('div', { class: 'placeholder-pane' },
        'no subtitles yet — upload an SRT in the Upload Center'));
      return;
    }
    const bulkSel = speakerSelect(null, k => {
      if (k === undefined) return;
      sel.forEach(n => setPending(n, k));
      sel.clear();                       // assignment done -> deselect rows
      render();
    }, '⇊ assign selected');
    bulkSel.disabled = false;

    const bar = el('div', { class: 'bar' },
      el('button', { class: 'primary', id: 'subs-save', onclick: save,
                     disabled: !pending.size }, 'Save'),
      el('button', { onclick: revert }, 'Revert to uploaded SRT'),
      el('button', { onclick: () => { location.href = `${P}/srt/export`; } },
        'Download SRT'),
      bulkSel);

    const allCb = el('input', { type: 'checkbox' });
    allCb.checked = sel.size && sel.size === rows.length;
    allCb.onchange = () => {
      sel.clear();
      if (allCb.checked) rows.forEach(r => sel.add(r.line_no));
      render();
    };

    const table = el('table', { class: 'subs-table' },
      el('thead', {}, el('tr', {},
        el('th', {}, allCb), el('th', {}, 'speaker'),
        el('th', {}, 'time'), el('th', {}, 'subtitle'))),
      el('tbody', {}, ...rows.map(r => rowEl(r))));

    root.replaceChildren(bar, table);
  }

  function currentSpk(r) {
    return pending.has(r.line_no) ? pending.get(r.line_no) : r.speaker;
  }

  function rowEl(r) {
    const cb = el('input', { type: 'checkbox' });
    cb.checked = sel.has(r.line_no);
    cb.onchange = () => { cb.checked ? sel.add(r.line_no) : sel.delete(r.line_no); };

    const spkSel = speakerSelect(currentSpk(r), k => {
      if (k === undefined) return;
      setPending(r.line_no, k);
      render();
    });

    return el('tr', { class: pending.has(r.line_no) ? 'dirty' : '' },
      el('td', {}, cb),
      el('td', {}, spkSel),
      el('td', { class: 'ts' }, `${fmtTs(r.start)}–${fmtTs(r.end)}`),
      el('td', { class: 'txt' }, r.text));
  }

  /* Build a speaker <select>. onPick receives the chosen key (null for "—",
     undefined for cancelled add). */
  function speakerSelect(current, onPick, placeholder) {
    const s = el('select');
    if (placeholder) s.append(el('option', { value: '__ph' }, placeholder));
    s.append(el('option', { value: '' }, '—'));
    for (const [k, name] of Object.entries(speakers)) {
      s.append(el('option', { value: k }, name));
    }
    s.append(el('option', { value: ADD }, '＋ add speaker'));
    if (!placeholder) s.value = current ?? '';
    s.onchange = async () => {
      if (s.value === ADD) {
        const k = await addAutoSpeaker();
        onPick(k);  // undefined if creation failed
      } else if (s.value === '__ph') {
        onPick(undefined);
      } else {
        onPick(s.value === '' ? null : s.value);
      }
    };
    return s;
  }

  function setPending(lineNo, spk) {
    const row = rows.find(r => r.line_no === lineNo);
    if ((row.speaker ?? null) === spk) pending.delete(lineNo);
    else pending.set(lineNo, spk);
  }

  /* speaker1, speaker2, … — first name not already taken */
  async function addAutoSpeaker() {
    const taken = new Set([...Object.keys(speakers),
                           ...Object.values(speakers)]);
    let i = 1;
    while (taken.has(`speaker${i}`)) i++;
    try {
      const out = await api.postJSON(`${P}/speakers`, { name: `speaker${i}` });
      speakers[out.key] = out.display_name;
      window.speakerCenter?.reload();
      return out.key;
    } catch (e) { alert(e.message); return undefined; }
  }

  /* ------------------------------ actions ------------------------------ */

  async function save() {
    if (!pending.size) return;
    const edits = [...pending].map(([line_no, value]) =>
      ({ line_no, field: 'speaker', value }));
    try {
      await api.send('PATCH', `${P}/dub/lines`, { edits });
      await reload();
      window.speakerCenter?.reload();
      window.dubCenter?.reload();
      window.uploadCenter?.refresh();
    } catch (e) { alert(e.message); }
  }

  async function revert() {
    if (!confirm('Revert to the originally uploaded SRT? This rebuilds the ' +
                 'line table from scratch: speaker assignments, overrides, ' +
                 'and all dub progress are reset; extracted clips and ' +
                 'generated takes are removed.')) return;
    try {
      await api.postJSON(`${P}/srt/revert`, {});
      await reload();
      window.refreshPanes?.();
    } catch (e) { alert(e.message); }
  }

  window.subCenter = { reload, isDirty };
  window.addEventListener('beforeunload', ev => {
    if (isDirty()) { ev.preventDefault(); ev.returnValue = ''; }
  });
  reload().catch(e => root.replaceChildren(
    el('div', { class: 'uc-status err' }, e.message)));
})();

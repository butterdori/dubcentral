/* Dub Center per-line grid.*/

(() => {
  const root = $('#dc-body');
  let data = null;
  const sel = new Set();
  let playing = null;
  let lastStats = '';

  const HINT = {
    exag: 'exaggeration: emotion intensity (0 flat … ~1 dramatic)',
    cfg: 'cfg_weight: how closely the voice follows the reference clips',
    tol: 'tolerance: Auto fit allows duration up to slot×tol (and down to slot÷tol) before stretching',
    fit: 'fit mode — auto: stretch into the subtitle slot when outside tolerance · natural: play at generated length (may overlap; mixed at remix) · manual: apply fac directly',
    fac: 'manual factor (Manual fit mode only): >1 speeds up, <1 slows down; ~0.7–1.5 recommended',
    ratio: 'placed (time-fitted) duration ÷ subtitle slot — after auto/manual stretching; hover for the pre-fit generated length',
    lang: 'dub language code (e.g. ko, ja, en)',
    speed: 'CosyVoice native speed knob: the model GENERATES at this pace (full regen). Time-fit/atempo still applies on top.',
    instr: 'CosyVoice style instruction in plain words, e.g. "speak with quiet urgency" — empty = plain voice cloning',
    crispasrBackend: 'which of CrispASR\'s hosted TTS backends to use for this line (e.g. "cosyvoice3"). CrispASR runs as a separate server.',
    orig: 'original-language line text — the reference clip\'s transcript for CosyVoice voice cloning (upload the original SRT to fill these)',
    engine: 'per-project synthesis engine. Switching needs to redub all lines.',
  };

  const BADGE = {
    never:       ['–',  'never dubbed'],
    clean:       ['✓',  'dubbed, up to date'],
    needs_light: ['≈',  'timing changed — light regen (refit only)'],
    needs_full:  ['↻',  'content changed — full regen (new TTS)'],
    generating:  ['…',  'generating'],
    failed:      ['⚠', 'failed'],
  };

  async function reload() {
    data = await api.get(`${P}/dub/grid`);
    for (const n of [...sel]) {
      if (!data.rows.some(r => r.line_no === n)) sel.delete(n);
    }
    render();
  }

  /* ------------------------------ render ------------------------------ */

  function render() {
    if (!data.rows.length) {
      root.replaceChildren(el('div', { class: 'placeholder-pane' },
        'no lines yet — upload an SRT in the Upload Center'));
      return;
    }
    const bar = el('div', { class: 'bar' },
      el('button', { class: 'primary', onclick: () => dub('all') }, 'Dub All'),
      el('button', { onclick: () => dub('selected'), disabled: !sel.size },
        `Dub Selected${sel.size ? ` (${sel.size})` : ''}`),
      el('button', { onclick: dubChanged, title:
        'regenerate only lines whose existing take went stale after edits ' +
        '(≈ light / ↻ full); never-dubbed and failed lines are left alone' },
        'Dub Changed'),
      (window.__dcTier = el('select', { class: 'mono', title:
        'which changed tier to regenerate' },
        el('option', { value: '' }, 'both'),
        el('option', { value: 'light' }, '≈ light only'),
        el('option', { value: 'full' }, '↻ full only'))),
      el('button', { onclick: undo, disabled: !data.has_undo }, 'Undo'),
      el('button', { onclick: remix, title:
        'splice dubbed windows onto the original audio and mux the video' },
        'Final Remix'),
      el('button', { onclick: () => { location.href = `${P}/remix/download`; },
        disabled: !data.has_final,
        title: data.has_final ? 'download the last remixed video'
                              : 'run Final Remix first' }, '⬇ Final'),
      el('span', { class: 'mono dim', id: 'dc-status' }, lastStats));

    root.replaceChildren(bar, optionsBar(), bulkBar(),
      el('div', { class: 'dc-wrap' }, table()));
  }

  function optionsBar() {
    const cb = el('input', { type: 'checkbox', id: 'dc-ts-mute' });
    cb.checked = !!data.timestamp_mute;
    cb.onchange = async () => {
      const want = cb.checked;
      try {
        const out = await api.send('PUT', `${P}/dub/timestamp_mute`, { value: want });
        data.timestamp_mute = out.timestamp_mute;
        status(`timestamp mute ${out.timestamp_mute ? 'on' : 'off'} — ` +
               (out.timestamp_mute
                 ? 'original audio muted for the full subtitle timestamp, even if the dub is shorter'
                 : 'original audio muted only for the dub\'s own duration'));
      } catch (e) { cb.checked = !want; status(e.message, true); }
    };
    const eng = el('select', { class: 'mono', title: HINT.engine +
      ' crispasr runs as a SEPARATE server process — this app has no ' +
      'visibility into or control over its GPU/VRAM usage (independent of ' +
      'the Force CPU checkbox, which does nothing for this engine unless ' +
      'you\'ve configured a second CPU-only instance).' },
      el('option', { value: 'chatterbox' }, 'engine: chatterbox'),
      el('option', { value: 'cosyvoice3' }, 'engine: cosyvoice3'),
      el('option', { value: 'crispasr' }, 'engine: crispasr'));
    eng.value = data.engine;
    eng.onchange = async () => {
      const want = eng.value;
      if (want === data.engine) return;
      const crispWarn = want === 'crispasr'
        ? '\n\ncrispasr runs as a separate server process this app doesn\'t ' +
          'manage — make sure it\'s actually running at the configured URL, ' +
          'and note the Force CPU checkbox has no effect on it unless a ' +
          'second CPU-only instance is set up.'
        : '';
      if (!confirm(`Switch engine to ${want}?\n\nEvery dubbed line flips to ` +
                   'full-regen (different engine = different audio), and the ' +
                   'other engine\'s knob overrides and speaker defaults are ' +
                   'cleared. Timing, text, speakers, and clips are untouched.' +
                   crispWarn)) {
        eng.value = data.engine;
        return;
      }
      try {
        const out = await api.send('PUT', `${P}/engine`, { engine: want });
        status(`engine: ${out.engine} — ${Object.keys(out.deltas).length} lines flagged for regen`);
        await reload();
        window.speakerCenter?.reload();
      } catch (e) { eng.value = data.engine; status(e.message, true); }
    };
    const cudaCb = el('input', { type: 'checkbox', id: 'dc-force-cpu' });
    cudaCb.checked = !data.cuda_enabled;   // checkbox reads as "Force CPU"
    cudaCb.onchange = async () => {
      const wantEnabled = !cudaCb.checked;
      try {
        const out = await api.send('PUT', `${P}/dub/cuda`, { value: wantEnabled });
        data.cuda_enabled = out.cuda_enabled;
        status(out.cuda_enabled ? 'CUDA enabled — GPU used for dubbing when available'
                                : 'CUDA disabled — TTS/dubbing forced to CPU (Demucs still uses GPU)');
      } catch (e) { cudaCb.checked = !wantEnabled; status(e.message, true); }
    };
    return el('div', { class: 'bar dc-options' },
      el('span', { class: 'dim mono' }, 'Options:'),
      eng,
      el('label', { class: 'dc-opt',
        title: 'Mute the original audio for the whole subtitle timestamp ' +
               'at remix time, not just the dub\'s own (post-fit) duration. ' +
               'Turn this on if a shorter dub is letting the tail of the ' +
               'original speech play through.' },
        cb, ' Timestamp mute'),
      el('label', { class: 'dc-opt',
        title: 'Force TTS generation (dubbing) onto CPU for this project, ' +
               'regardless of CUDA availability — a troubleshooting ' +
               'override for GPU/VRAM issues. Demucs separation (Speaker ' +
               'Center extraction, and Final Remix) always uses GPU when ' +
               'available and is unaffected. Takes effect on the next dub ' +
               'run, never mid-run.' },
        cudaCb, ' Force CPU (dubbing only)'));
  }

  /* ---- bulk-set bar: apply fields to every selected line in one PATCH ---- */
  function bulkBar() {
    const spk = el('select', { title: 'set speaker for selected lines' },
      el('option', { value: '' }, 'spk…'),
      el('option', { value: '__none' }, '— none'),
      ...Object.entries(data.speakers).map(([k, n]) => el('option', { value: k }, n)));
    const engine = data.engine;
    const cosy = engine === 'cosyvoice3';
    const crisp = engine === 'crispasr';
    const lang = el('input', { type: 'text', placeholder: 'lang', title: HINT.lang });
    const exag = crisp
      ? el('input', { type: 'text', placeholder: 'backend', title: HINT.crispasrBackend })
      : el('input', { type: 'number', step: '0.05',
      placeholder: cosy ? 'spd' : 'exag', title: cosy ? HINT.speed : HINT.exag });
    const cfg  = el('input', { type: cosy ? 'text' : 'number', step: '0.05',
      placeholder: cosy ? 'instr' : 'cfg', title: cosy ? HINT.instr : HINT.cfg,
      hidden: crisp });
    const clearInstr = el('input', { type: 'checkbox',
      title: 'Clear instr (revert to inherited/blank) for selected lines, ' +
             'instead of setting new text — the text box alone can\'t ' +
             'express "erase everything" since a blank box normally just ' +
             'means "leave this field alone".' });
    clearInstr.onchange = () => { if (clearInstr.checked) cfg.value = ''; cfg.disabled = clearInstr.checked; };
    cfg.oninput = () => { if (cfg.value) { clearInstr.checked = false; cfg.disabled = false; } };
    const tol  = el('input', { type: 'number', step: '0.05', placeholder: 'tol',  title: HINT.tol });
    const fit = el('select', { title: HINT.fit },
      el('option', { value: '' }, 'fit…'),
      el('option', { value: '__inherit' }, '↩ inherit'),
      ...['auto', 'natural', 'manual'].map(m => el('option', { value: m }, m)));
    const fac = el('input', { type: 'number', step: '0.1', placeholder: 'fac', title: HINT.fac });

    const apply = el('button', { disabled: !sel.size, onclick: async () => {
      const per = [];   // field/value pairs to fan out
      if (spk.value) per.push(['speaker', spk.value === '__none' ? null : spk.value]);
      if (lang.value.trim()) per.push(['dub_language', lang.value.trim()]);
      if (crisp) {
        if (exag.value.trim()) per.push(['crispasr_backend', exag.value.trim()]);
      } else {
      const numPairs = cosy
        ? [[exag, 'speed'], [tol, 'tolerance'], [fac, 'manual_factor']]
        : [[exag, 'exaggeration'], [cfg, 'cfg_weight'],
           [tol, 'tolerance'], [fac, 'manual_factor']];
      for (const [inp, f] of numPairs) {
        if (inp.value !== '') per.push([f, Number(inp.value)]);
        }
      }
      if (crisp) {
        if (tol.value !== '') per.push(['tolerance', Number(tol.value)]);
        if (fac.value !== '') per.push(['manual_factor', Number(fac.value)]);
      }
      if (cosy) {
        if (clearInstr.checked) per.push(['instruct_text', null]);
        else if (cfg.value.trim() !== '') per.push(['instruct_text', cfg.value.trim()]);
      }
      if (fit.value) per.push(['fit_mode', fit.value === '__inherit' ? null : fit.value]);
      if (!per.length) { status('bulk: nothing filled in', true); return; }
      const edits = [];
      for (const n of sel) for (const [field, value] of per) edits.push({ line_no: n, field, value });
      try {
        await api.send('PATCH', `${P}/dub/lines`, { edits });
        status(`bulk applied ${per.map(x => x[0]).join(', ')} to ${sel.size} lines`);
        await reload();
        if (per.some(x => x[0] === 'speaker')) window.speakerCenter?.reload();
      } catch (e) { status(e.message, true); reload(); }
    } }, 'Apply to selected');

    const cfgGroup = cosy
      ? el('span', { class: 'dc-instr-group' }, cfg,
          el('label', { class: 'dc-clear-instr', title: clearInstr.title },
            clearInstr, ' clear'))
      : cfg;

    return el('div', { class: 'bar dc-bulk' },
      el('span', { class: 'dim mono' }, 'set selected:'),
      spk, lang, exag, cfgGroup, tol, fit, fac, apply);
  }

  function table() {
    const allCb = el('input', { type: 'checkbox' });
    allCb.checked = sel.size && sel.size === data.rows.length;
    allCb.onchange = () => {
      sel.clear();
      if (allCb.checked) data.rows.forEach(r => sel.add(r.line_no));
      render();
    };
    return el('table', { class: 'dc-table' },
      el('thead', {}, el('tr', {},
        el('th', {}, allCb), el('th', {}, '#'),
        el('th', {}, 'start'), el('th', {}, 'end'),
        el('th', {}, 'speaker'), el('th', {}, 'text'),
        el('th', { title: HINT.lang }, 'lang'),
        ...(data.engine === 'cosyvoice3'
          ? [el('th', { title: HINT.orig }, 'orig text'),
             el('th', { title: HINT.speed }, 'spd'),
             el('th', { title: HINT.instr }, 'instr')]
          : data.engine === 'crispasr'
          ? [el('th', { title: HINT.orig }, 'orig text'),
             el('th', { title: HINT.crispasrBackend }, 'backend')]
          : [el('th', { title: HINT.exag }, 'exag'),
             el('th', { title: HINT.cfg }, 'cfg')]),
        el('th', { title: HINT.tol }, 'tol'),
        el('th', { title: HINT.fit }, 'fit'),
        el('th', { title: HINT.fac }, 'fac'),
        el('th', { title: HINT.ratio }, 'ratio'),
        el('th', {}, 'st'), el('th', {}, ''))),
      el('tbody', {}, ...data.rows.map(r => rowEl(r))));
  }

  /* inheritable field input: effective value shown AS the value; purple
     outline = line override; empty -> clear override (re-inherit) */
  function numInput(r, field, opts = {}) {
    const f = r.fields[field];
    return el('input', {
      type: 'number', step: opts.step || '0.05',
      class: f.source === 'line' ? 'ovr' : '',
      value: f.value,
      title: `${field}: ${f.value} (from ${f.source})` +
             (f.source === 'line' ? ' — clear to re-inherit' : ''),
      onchange: ev => edit(r.line_no, field,
        ev.target.value === '' ? null : Number(ev.target.value)),
    });
  }

  /* original-language transcript cell — shared by cosyvoice3 and crispasr */
  function origTextCell(r) {
    return el('td', { class: 'txt orig' }, el('input', { type: 'text',
      value: r.original_text ?? '', placeholder: '(no original text)',
      title: HINT.orig,
      onchange: ev => edit(r.line_no, 'original_text',
        ev.target.value.trim() === '' ? null : ev.target.value) }));
  }

  function rowEl(r) {
    const cb = el('input', { type: 'checkbox' });
    cb.checked = sel.has(r.line_no);
    cb.onchange = () => {
      cb.checked ? sel.add(r.line_no) : sel.delete(r.line_no);
      render();
    };

    const spkSel = el('select', { class: 'spk' },
      el('option', { value: '' }, '—'),
      ...Object.entries(data.speakers).map(([k, name]) =>
        el('option', { value: k }, name)));
    spkSel.value = r.speaker ?? '';
    spkSel.onchange = () => edit(r.line_no, 'speaker',
      spkSel.value === '' ? null : spkSel.value);

    const langF = r.fields.dub_language;
    const lang = el('input', {
      type: 'text', class: langF.source === 'line' ? 'ovr' : '',
      value: langF.value,
      title: `language: ${langF.value} (from ${langF.source})` +
             (langF.source === 'line' ? ' — clear to re-inherit' : ''),
      onchange: ev => edit(r.line_no, 'dub_language',
        ev.target.value.trim() === '' ? null : ev.target.value.trim()),
    });

    const fitF = r.fields.fit_mode;
    const fit = el('select', { class: 'fit' + (fitF.source === 'line' ? ' ovr' : ''),
                               title: HINT.fit + ` — current: ${fitF.value} (from ${fitF.source})` },
      ...['auto', 'natural', 'manual'].map(m => el('option', { value: m }, m)),
      el('option', { value: '__inherit' }, '↩ inherit'));
    fit.value = fitF.value;
    fit.onchange = () => edit(r.line_no, 'fit_mode',
      fit.value === '__inherit' ? null : fit.value);

    const placed = r.fit_duration_s ?? r.raw_duration_s;   // placed length
    const ratio = placed && r.slot_s ? placed / r.slot_s : null;
    const [glyph, hint] = BADGE[r.badge] || ['?', r.badge];
    const badge = el('span', { class: `badge b-${r.badge}`,
      title: r.badge === 'failed' ? `failed: ${r.error}` : hint }, glyph);

    const play = el('button', { class: 'dc-play', title: 'Play dubbed line' }, '▶');
    play.onclick = () => togglePlay(r.line_no, play);
    play.disabled = ['never', 'failed'].includes(r.badge);

    return el('tr', { dataset: { line: r.line_no } },
      el('td', {}, cb),
      el('td', { class: 'mono dim' }, String(r.line_no)),
      el('td', { class: 'w-num' }, el('input', { type: 'number', step: '0.001',
        value: r.start.toFixed(3),
        onchange: ev => edit(r.line_no, 'start', Number(ev.target.value)) })),
      el('td', { class: 'w-num' }, el('input', { type: 'number', step: '0.001',
        value: r.end.toFixed(3),
        onchange: ev => edit(r.line_no, 'end', Number(ev.target.value)) })),
      el('td', {}, spkSel),
      el('td', { class: 'txt' }, el('input', { type: 'text', value: r.text,
        onchange: ev => edit(r.line_no, 'text', ev.target.value) })),
      el('td', { class: 'w-lang' }, lang),
      ...(data.engine === 'cosyvoice3'
        ? [origTextCell(r),
           el('td', { class: 'w-sm' }, numInput(r, 'speed', { step: '0.05' })),
           el('td', { class: 'w-instr' }, (() => {
             const f = r.fields.instruct_text;
             return el('input', { type: 'text',
               class: f.source === 'line' ? 'ovr' : '',
               value: f.value, title: `${HINT.instr} (from ${f.source})`,
               onchange: ev => edit(r.line_no, 'instruct_text',
                 ev.target.value.trim() === '' ? null : ev.target.value.trim()) });
           })())]
        : data.engine === 'crispasr'
        ? [origTextCell(r),
           el('td', { class: 'w-instr' }, (() => {
             const f = r.fields.crispasr_backend;
             return el('input', { type: 'text',
               class: f.source === 'line' ? 'ovr' : '',
               value: f.value, title: `${HINT.crispasrBackend} (from ${f.source})`,
               onchange: ev => edit(r.line_no, 'crispasr_backend',
                 ev.target.value.trim() === '' ? null : ev.target.value.trim()) });
           })())]
        : [el('td', { class: 'w-sm' }, numInput(r, 'exaggeration')),
           el('td', { class: 'w-sm' }, numInput(r, 'cfg_weight'))]),
      el('td', { class: 'w-sm' }, numInput(r, 'tolerance')),
      el('td', {}, fit),
      el('td', { class: 'w-sm' }, numInput(r, 'manual_factor', { step: '0.1' })),
      el('td', { class: 'ratio' + (ratio && (ratio > 2 || ratio < 0.5) ? ' hard' : ''),
                 title: ratio
                   ? `placed ${placed.toFixed(2)}s / slot ${r.slot_s.toFixed(2)}s` +
                     (r.raw_duration_s ? ` (generated ${r.raw_duration_s.toFixed(2)}s)` : '') +
                     ` — ${HINT.ratio}`
                   : HINT.ratio },
        ratio ? `${ratio.toFixed(2)}×` : '–'),
      el('td', {}, badge),
      el('td', {}, play));
  }

  /* ------------------------------ edits ------------------------------- */

  async function edit(lineNo, field, value) {
    try {
      const out = await api.send('PATCH', `${P}/dub/lines`,
        { edits: [{ line_no: lineNo, field, value }] });
      for (const row of out.rows) {
        const i = data.rows.findIndex(r => r.line_no === row.line_no);
        if (i >= 0) data.rows[i] = row;
      }
      render();
      if (field === 'speaker') window.speakerCenter?.reload();
    } catch (e) {
      alert(e.message);
      reload();
    }
  }

  /* ------------------------------- runs ------------------------------- */

  async function dub(mode) {
    try {
      const body = mode === 'all' ? { mode } : { mode, line_nos: [...sel] };
      const out = await api.send('POST', `${P}/dub/run`, body);
      status(out.appended
        ? `added ${out.n_targets} lines to the running dub…`
        : `queued ${out.n_targets} lines…`);
    } catch (e) { status(e.message, true); }
  }

  async function dubChanged() {
    try {
      const tier = window.__dcTier?.value || null;
      const out = await api.send('POST', `${P}/dub/run`,
        { mode: 'changed', tier });
      status(`queued ${out.n_targets} changed lines…`);
    } catch (e) { status(e.message, true); }
  }

  async function remix() {
    try {
      await api.send('POST', `${P}/remix`, {});
      status('remix queued…');
    } catch (e) { status(e.message, true); }
  }

  async function undo() {
    try {
      const out = await api.send('POST', `${P}/dub/undo`, {});
      status(`undid last dub action (${out.restored.length} lines)`);
      await reload();
    } catch (e) { status(e.message, true); }
  }

  function status(msg, isErr = false) {
    lastStats = msg;
    const n = $('#dc-status');
    if (n) { n.textContent = msg; n.classList.toggle('err', isErr); }
  }

  function togglePlay(lineNo, btn) {
    if (playing) {
      playing.audio.pause();
      playing.btn.textContent = '▶';
      const same = playing.btn === btn;
      playing = null;
      if (same) return;
    }
    const audio = new Audio(`${P}/dub/lines/${lineNo}/audio?ts=${Date.now()}`);
    audio.onended = () => { btn.textContent = '▶'; playing = null; };
    audio.onerror = () => { btn.textContent = '▶'; playing = null; };
    audio.play();
    btn.textContent = '⏸';
    playing = { audio, btn };
  }

  /* --------------------- live updates via polling ---------------------- */

  const reportedJobIds = new Set();
  let seenFirstPoll = false;
  window.jobWatch.on(s => {
    const j = s.job;
    /* first-poll seeding BEFORE the kind filter — see speaker-center.js
       for why the other order silently swallows fast fresh jobs */
    if (!seenFirstPoll) {
      seenFirstPoll = true;
      if (j && j.status !== 'running' && j.status !== 'queued') {
        reportedJobIds.add(j.id);
      }
    }
    if (!j || j.project_key !== KEY) return;
    if (j.kind !== 'dub' && j.kind !== 'remix') return;
    const active = j.status === 'running' || j.status === 'queued';
    if (active) {
      status(j.message || j.status);
      if (j.kind === 'dub') applyDeltas(j.deltas);
      return;
    }
    if (reportedJobIds.has(j.id)) return;
    reportedJobIds.add(j.id);
    {
      if (j.status === 'failed') {
        status(`${j.kind} job failed: ${j.error}`, true);
      } else if (j.kind === 'remix') {
        const r = j.result;
        status(`remix done · ${r.n_lines_spliced} lines · ${r.elapsed_s}s · ` +
               `output ${r.output_duration_s}s`);
      } else {
        const r = j.result;
        const bits = [`ok ${r.n_ok}`];
        if (r.n_failed) bits.push(`failed ${r.n_failed}`);
        bits.push(`${r.elapsed_s}s`);
        if (r.raw_vs_slot_ratio != null) bits.push(`dub audio ${r.raw_vs_slot_ratio}× slot time`);
        if (r.hard_stretched_lines?.length) bits.push(`hard-stretched: ${r.hard_stretched_lines.join(',')}`);
        if (r.n_failed && r.errors) {
          const [n, msg] = Object.entries(r.errors)[0];
          bits.push(`line ${n}: ${msg}`);
        }
        status(bits.join(' · '), r.n_failed > 0);
      }
      reload();
    }
  });

  function applyDeltas(deltas) {
    let dirty = false;
    for (const [n, badge] of Object.entries(deltas)) {
      const row = data?.rows.find(r => r.line_no === Number(n));
      if (row && row.badge !== badge) { row.badge = badge; dirty = true; }
    }
    if (dirty) render();
  }

  window.dubCenter = { reload };
  reload().catch(e => root.replaceChildren(
    el('div', { class: 'uc-status err' }, e.message)));
})();

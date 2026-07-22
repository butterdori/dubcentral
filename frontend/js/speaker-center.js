/* Speaker Center: extraction, speaker cards (rename, inherit values), prune*/

(() => {
  const root = $('#sc-body');
  const FIELDS_BY_ENGINE = {
    chatterbox: [
      ['dub_language', 'lang', 'text'],
      ['exaggeration', 'exag', 'number'],
      ['cfg_weight', 'cfg', 'number'],
      ['tolerance', 'tol', 'number'],
    ],
    cosyvoice3: [
      ['dub_language', 'lang', 'text'],
      ['speed', 'spd', 'number'],
      ['instruct_text', 'instr', 'text'],
      ['tolerance', 'tol', 'number'],
    ],
    crispasr: [
      ['dub_language', 'lang', 'text'],
      ['crispasr_backend', 'backend', 'text'],
      ['instruct_text', 'instr', 'text'],
      ['tolerance', 'tol', 'number'],
    ],
  };
  let DEFAULT_FIELDS = FIELDS_BY_ENGINE.chatterbox;
  let ENGINE = 'chatterbox';
  const REF_PIN_ENGINES = new Set(['cosyvoice3', 'crispasr']);

  let playing = null;   // {audio, btn}
  const expanded = new Set();   // speaker keys with clip list open
  const testText = new Map();

  /* ---------- shared job polling: one loop, many listeners ---------- */
  const jobWatch = {
    listeners: new Set(),
    timer: null,
    start() {
      if (this.timer) return;
      const tick = async () => {
        let s = null;
        try { s = await api.get('/api/jobs/current'); }
        catch (e) { /* server briefly unreachable; keep polling */ }
        if (s) this.listeners.forEach(fn => {
          try { fn(s); }
          catch (e) { console.error('jobWatch listener error:', e); }
        });
        const active = s && s.job && ['queued', 'running'].includes(s.job.status);
        this.timer = setTimeout(tick, active ? 700 : 2500);
      };
      tick();
    },
    on(fn) { this.listeners.add(fn); this.start(); },
  };
  window.jobWatch = jobWatch;

  /* ---------------------------- rendering ---------------------------- */

  async function reload() {
    const data = await api.get(`${P}/speakers`);
    render(data);
  }

  function render({ speakers, project_defaults, engine, has_original_text }) {
    DEFAULT_FIELDS = FIELDS_BY_ENGINE[engine] || FIELDS_BY_ENGINE.chatterbox;
    ENGINE = engine;
    const warn = (REF_PIN_ENGINES.has(engine) && !has_original_text)
      ? el('div', { class: 'sc-warn' },
          '⚠ No original-language text uploaded — CosyVoice needs a ' +
          'transcript of each reference clip to clone a voice. Upload the ' +
          'original SRT in the Upload Center.')
      : '';
    const anyDirty = REF_PIN_ENGINES.has(engine) && speakers.some(s => s.ref_dirty);
    root.replaceChildren(
      el('div', { class: 'bar' },
        el('button', { id: 'btn-extract', onclick: startExtraction },
          'Extract reference clips'),
        el('button', { onclick: addSpeaker }, 'Add speaker'),
        ...(REF_PIN_ENGINES.has(engine) ? [
          el('button', { onclick: renewReference,
            class: anyDirty ? 'primary' : '',
            title: 'Rebuild each speaker\'s CosyVoice reference from their ' +
                   'currently pinned (📌) clips, concatenated into one file. ' +
                   'Speakers with no clips pinned fall back to auto-picking ' +
                   'a single best clip.' },
            'Renew reference' + (anyDirty ? ' •' : ''))
        ] : []),
        el('span', { class: 'sc-job mono dim', id: 'sc-job' })),
      warn,
      speakers.length
        ? el('div', { class: 'sc-list' }, ...speakers.map(s => card(s)))
        : el('div', { class: 'placeholder-pane' },
            'no speakers yet — assign them in segments.csv, or add one above'));
  }

  function card(s) {
    const open = expanded.has(s.key);
    const toggle = () => {
      open ? expanded.delete(s.key) : expanded.add(s.key);
      reload();
    };
    const caret = el('button', { class: 'caret',
      title: open ? 'Collapse clips' : 'Expand clips',
      onclick: toggle }, open ? '▾' : '▸');
    const fields = DEFAULT_FIELDS.map(([field, label, type]) => {
      const explicit = field in s.defaults;
      const inp = el('input', {
        type, step: '0.05', class: explicit ? 'explicit' : '',
        value: explicit ? s.defaults[field] : '',
        placeholder: String(s.effective[field].value),
        title: explicit ? `set on this speaker (clear to inherit)`
                        : `inherited from project (${s.effective[field].value})`,
        onchange: ev => setDefault(s.key, field, type, ev.target.value),
      });
      return el('label', { class: 'sc-field' },
        el('span', { class: 'mono dim' }, label), inp);
    });
    const head = el('div', { class: 'sc-head' },
      caret,
      s.clips.length ? el('span', { class: 'mono has-clips',
        title: `reference clips extracted (${s.clips.length})` }, '✓') : '',
      el('input', { class: 'sc-name', type: 'text', value: s.display_name,
        title: 'Rename speaker (display only)',
        onchange: ev => rename(s.key, ev.target.value) }),
      ...fields,
      (REF_PIN_ENGINES.has(ENGINE) && s.ref_dirty)
        ? el('span', { class: 'mono warn-pin',
            title: 'Pinned clips have changed since the reference was last ' +
                   'renewed — click "Renew reference" above to rebuild it ' +
                   'and apply the change.' }, '● unrenewed')
        : '',
      el('span', { class: 'mono dim meta', onclick: toggle,
        title: open ? 'Collapse clips' : 'Expand clips' },
        `${s.n_lines} lines · ${s.clips.length} clips` +
        (s.pruned.length ? ` · ${s.pruned.length} pruned` : '')));

    const parts = [head];
    if (open) {
      const clips = el('div', { class: 'sc-clips' },
        ...s.clips.map(c => clipRow(s, c)));
      if (!s.clips.length) {
        clips.append(el('span', { class: 'dim mono' }, 'no clips extracted'));
      }
      parts.push(testSpeechRow(s));
      parts.push(clips);
    }
    return el('div', { class: 'sc-card' }, ...parts);
  }

  function testSpeechRow(s) {
    const input = el('input', { type: 'text', class: 'sc-test-input',
      value: testText.get(s.key) || '',
      placeholder: 'type a line to test this voice…',
      title: 'Synthesizes with this speaker\'s CURRENT effective reference ' +
             'and knobs — the same ones a real dub would use — so you can ' +
             'sanity-check the voice without dubbing a real line.',
    oninput: () => testText.set(s.key, input.value) });
    const genBtn = el('button', { class: 'sc-test-gen' }, 'Generate');
    const playBtn = el('button', { class: 'sc-play', title: 'Play test speech',
      disabled: true }, '▶');
    const dlBtn = el('a', { class: 'sc-test-dl', title: 'Download test speech',
      href: '#', target: '_blank', style: 'pointer-events:none;opacity:.4' }, '⬇');

    function setHasAudio(has) {
      playBtn.disabled = !has;
      dlBtn.style.pointerEvents = has ? 'auto' : 'none';
      dlBtn.style.opacity = has ? '1' : '.4';
      dlBtn.href = has ? `${P}/speakers/${encodeURIComponent(s.key)}/test_speech/download` : '#';
    }
    setHasAudio(s.has_test_speech);

    genBtn.onclick = async () => {
      const text = input.value.trim();
      if (!text) return;
      genBtn.disabled = true;
      genBtn.textContent = 'Generating…';
      try {
        await api.postJSON(`${P}/speakers/${encodeURIComponent(s.key)}/test_speech`,
          { text });
        status(`generating test speech for ${s.display_name}…`);
      } catch (e) { status(e.message, true); genBtn.disabled = false; genBtn.textContent = 'Generate'; }
    };
    playBtn.onclick = () => togglePlay(s.key, '__test', playBtn,
      `${P}/speakers/${encodeURIComponent(s.key)}/test_speech/audio`);

    return el('div', { class: 'sc-test-row' },
      input, genBtn, playBtn, dlBtn);
  }

  function clipRow(s, c) {
    const play = el('button', { class: 'sc-play', title: 'Play clip' }, '▶');
    play.onclick = () => togglePlay(s.key, c.line_no, play);
    const del = el('button', { class: 'danger sc-del', title: 'Delete clip (prune)' }, '✕');
    del.onclick = () => pruneClip(s.key, c.line_no);

    const parts = [play];
    if (REF_PIN_ENGINES.has(ENGINE)) {
      const isPinned = s.ref_pins.includes(c.line_no);
      const pin = el('button', {
        class: 'sc-pin' + (isPinned ? ' active' : ''),
        title: isPinned
          ? 'Pinned — click to unpin. Pinned clips are concatenated into ' +
            'the reference on the next "Renew reference".'
          : 'Pin this clip — multiple pins are concatenated together into ' +
            'one reference on "Renew reference".',
      }, '📌');
      pin.onclick = () => toggleRefPin(s, c.line_no);
      parts.push(pin);
    }
    parts.push(
      el('span', { class: 'mono num' }, String(c.line_no).padStart(4, '0')),
      el('span', { class: 'clip-text', title: c.text }, c.text || '(no text)'),
      (REF_PIN_ENGINES.has(ENGINE) && c.is_reference)
        ? el('span', { class: 'mono active-tag',
            title: 'This clip is part of the reference currently used for synthesis' },
            'active')
        : '',
      el('span', { class: 'mono dur' },
        (c.duration_s != null ? `${c.duration_s.toFixed(2)}s` : '?') +
        (c.too_short ? ' short' : '')),
      del);
    return el('div', { class: 'sc-clip' + (c.too_short ? ' short' : '') }, ...parts);
  }

  async function toggleRefPin(s, lineNo) {
    const next = s.ref_pins.includes(lineNo)
      ? s.ref_pins.filter(n => n !== lineNo)
      : [...s.ref_pins, lineNo];
    try {
      await api.send(
        'PUT', `${P}/speakers/${encodeURIComponent(s.key)}/reference_pins`,
        { line_nos: next });
    } catch (e) { alert(e.message); }
    reload();
  }

  async function renewReference() {
    try {
      await api.postJSON(`${P}/renew_reference`, {});
      status('renewing references…');
    } catch (e) { status(e.message, true); }
  }


  /* ----------------------------- actions ----------------------------- */

  function togglePlay(spk, lineNo, btn, urlOverride) {
    if (playing) {
      playing.audio.pause();
      playing.btn.textContent = '▶';
      const same = playing.btn === btn;
      playing = null;
      if (same) return;
    }
    const url = urlOverride ||
      `${P}/speakers/${encodeURIComponent(spk)}/clips/${lineNo}`;
    const audio = new Audio(url + (url.includes('?') ? '&' : '?') + `ts=${Date.now()}`);
    audio.onended = () => { btn.textContent = '▶'; playing = null; };
    audio.onerror = () => { btn.textContent = '▶'; playing = null; };
    audio.play();
    btn.textContent = '⏸';
    playing = { audio, btn };
  }

  async function pruneClip(spk, lineNo) {
    if (!confirm(`Delete clip line ${lineNo} from ${spk}? Re-extraction will ` +
                 'not bring it back, and the concatenated reference rebuilds ' +
                 'without it.')) return;
    try { await api.del(`${P}/speakers/${encodeURIComponent(spk)}/clips/${lineNo}`); }
    catch (e) { alert(e.message); }
    reload();
  }

  async function rename(spk, name) {
    try {
      await fetch(`${P}/speakers/${encodeURIComponent(spk)}`, {
        method: 'PATCH', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ display_name: name }),
      }).then(r => api._handle(r));
    } catch (e) { alert(e.message); }
    reload();
  }

  async function setDefault(spk, field, type, raw) {
    const value = raw === '' ? null : (type === 'number' ? Number(raw) : raw);
    try {
      const out = await api.putText(
        `${P}/speakers/${encodeURIComponent(spk)}/defaults`,
        JSON.stringify({ field, value }), 'application/json');
      const n = Object.keys(out.deltas).length;
      if (n) status(`${n} line${n > 1 ? 's' : ''} flagged for regen`);
    } catch (e) { alert(e.message); }
    reload();
    window.dubCenter?.reload();   // grid shows inherited values -> refresh
  }

  async function addSpeaker() {
    const name = (prompt('New speaker name:') || '').trim();
    if (!name) return;
    try { await api.postJSON(`${P}/speakers`, { name }); }
    catch (e) { alert(e.message); }
    reload();
  }

  async function startExtraction() {
    if (window.subCenter?.isDirty() &&
        !confirm('The Sub Center has unsaved speaker assignments — ' +
                 'extraction uses the last SAVED state. Continue anyway?')) {
      return;
    }
    try {
      await api.postJSON(`${P}/extract_clips`, {});
      status('queued…');
    } catch (e) { status(e.message, true); }
  }

  function status(msg, isErr = false) {
    const n = $('#sc-job');
    if (!n) return;
    n.textContent = msg;
    n.classList.toggle('err', isErr);
  }

  /* ------------------------- job status wiring ------------------------ */
  /* Report a job's terminal status exactly once per job id, regardless of
     whether we ever observed it "active" first — a fast job (e.g.
     re-extracting clips that already exist) can finish between our POST
     and the very next poll tick, and a transition-based check like "was it
     active, is it now done" would simply never fire, leaving the status
     line stuck on our own optimistic "queued…" text forever. */
  const reportedJobIds = new Set();
  let seenFirstPoll = false;
  jobWatch.on(s => {
    const j = s.job;
    /* First tick after page load: whatever finished job occupies the slot
       (ANY kind) predates this page view — mark it reported so it's never
       announced. This must happen BEFORE the kind filter: if it lived
       after, kind-filtered ticks wouldn't count as "seen", and the first
       MATCHING tick could be a job the user just started that finished
       within one poll interval — which would then be misclassified as
       pre-existing and silently swallowed (the "Renew stuck at
       'renewing references…'" bug). */
    if (!seenFirstPoll) {
      seenFirstPoll = true;
      if (j && j.status !== 'running' && j.status !== 'queued') {
        reportedJobIds.add(j.id);
      }
    }
    if (!j || j.project_key !== KEY) return;
    if (j.kind !== 'extract_clips' && j.kind !== 'renew_reference' &&
        j.kind !== 'test_speech') return;
    const active = j.status === 'running' || j.status === 'queued';
    if (active) { status(j.message || j.status); return; }
    if (reportedJobIds.has(j.id)) return;
    reportedJobIds.add(j.id);
    if (j.status === 'failed') {
      status(`job failed: ${j.error}`, true);
      reload();   // re-render so a stuck "Generate…" button resets
      return;
    }
    if (j.kind === 'extract_clips') {
      const per = j.result.per_speaker || {};
      const total = Object.values(per).reduce((a, b) => a + b, 0);
      status(`extracted ${total} clips ✓` +
             (j.result.skipped_pruned ? ` (${j.result.skipped_pruned} pruned kept out)` : ''));
    } else if (j.kind === 'test_speech') {
      status(`test speech ready for ${j.result.speaker} (${j.result.duration_s}s)`);
    } else {
      const n = Object.keys(j.result.deltas || {}).length;
      const modes = Object.values(j.result.speakers || {});
      const pinned = modes.filter(m => m.mode === 'pinned').length;
      const skipped = modes.reduce((a, m) => a + (m.skipped?.length || 0), 0);
      status(`references renewed — ${pinned} speaker${pinned === 1 ? '' : 's'} pinned` +
             (skipped ? `, ${skipped} stale pin${skipped === 1 ? '' : 's'} skipped` : '') +
             (n ? ` — ${n} line${n > 1 ? 's' : ''} flagged for regen` : ''));
      window.dubCenter?.reload();   // badges may have changed
    }
    reload();
  });

  window.speakerCenter = { reload };
  reload().catch(e => root.append(
    el('div', { class: 'uc-status err' }, e.message)));
})();

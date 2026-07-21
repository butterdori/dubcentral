const api = {
  async _handle(res) {
    if (res.ok) {
      const ct = res.headers.get('content-type') || '';
      return ct.includes('application/json') ? res.json() : res.text();
    }
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) msg = typeof body.detail === 'string'
        ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* non-JSON error body */ }
    throw new Error(msg);
  },

  get(path)  { return fetch(path).then(r => api._handle(r)); },
  del(path)  { return fetch(path, { method: 'DELETE' }).then(r => api._handle(r)); },
  postJSON(path, body) {
    return fetch(path, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    }).then(r => api._handle(r));
  },
  send(method, path, body) {
    return fetch(path, {
      method, headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    }).then(r => api._handle(r));
  },
  putText(path, text, mime = 'text/csv') {
    return fetch(path, {
      method: 'PUT', headers: { 'content-type': mime }, body: text,
    }).then(r => api._handle(r));
  },
  upload(path, file) {
    const fd = new FormData();
    fd.append('file', file);
    return fetch(path, { method: 'POST', body: fd }).then(r => api._handle(r));
  },

  // jobs polling helper
  // calls onTick(status) every
  // intervalMs until stop() is invoked or the endpoint reports idle.
  pollJobs(onTick, intervalMs = 1000) {
    let stopped = false;
    async function tick() {
      if (stopped) return;
      try {
        const s = await api.get('/api/jobs/current');
        onTick(s);
      } catch (e) { console.warn('jobs poll failed:', e.message); }
      if (!stopped) setTimeout(tick, intervalMs);
    }
    tick();
    return { stop() { stopped = true; } };
  },
};

/* tiny DOM helpers */
const $ = (sel, root = document) => root.querySelector(sel);
function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(n.dataset, v);
    else if (v === true) n.setAttribute(k, '');
    else n.setAttribute(k, v);
  }
  n.append(...children);
  return n;
}
function fmtDur(s) {
  if (s == null) return '—';
  s = Math.round(s);
  const m = Math.floor(s / 60), sec = String(s % 60).padStart(2, '0');
  return `${m}:${sec}`;
}

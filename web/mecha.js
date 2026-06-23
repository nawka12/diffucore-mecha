// Diffucore UI ⇄ sd-mecha — frontend.
//
// Renders a "Merge" tab whose method list, model slots, and hyperparameter
// fields are all built from whatever the backend introspects out of sd-mecha at
// runtime (GET /api/ext/diffucore-mecha/methods). Nothing about a specific merge
// method is hardcoded here, so new sd-mecha methods just appear.

(function () {
  const NAME = 'diffucore-mecha';
  const API = `/api/ext/${NAME}`;

  let METHODS = {};                 // id -> spec {models, varargs, params, doc}
  let FILES = { 'checkpoints': [], 'diffusion-models': [] };

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  async function loadMethods() {
    try {
      const r = await fetch(`${API}/methods`);
      const d = await r.json();
      METHODS = {};
      (d.methods || []).forEach((m) => { METHODS[m.id] = m; });
      return d;
    } catch (e) {
      return { installed: false, error: String(e), methods: [] };
    }
  }

  async function loadFiles() {
    try {
      const r = await fetch('/api/models');
      const d = await r.json();
      FILES = { 'checkpoints': d.checkpoints || [], 'diffusion-models': d.dits || [] };
    } catch (e) { /* leave empties; selects just show no options */ }
  }

  // ── tab ────────────────────────────────────────────────────────────────
  window.DiffucoreExt.registerTab({
    id: NAME,
    title: 'Merge',
    mount(el) {
      el.innerHTML = `
        <div style="max-width:720px">
          <h2 style="font-family:var(--serif);font-weight:400;margin:0 0 4px">Model <em style="color:var(--accent)">merge</em></h2>
          <p class="hint" style="margin:0 0 14px">Powered by <a href="https://github.com/ljleb/sd-mecha" target="_blank" rel="noopener">sd-mecha</a>.
            Methods and their parameters are read from sd-mecha at runtime.</p>
          <div id="mecha-body"></div>
        </div>`;
      const body = el.querySelector('#mecha-body');

      let myJobId = null;

      // Live progress / completion via the shared SSE stream.
      const es = new EventSource('/api/events');
      el._mechaEs = es;
      es.onmessage = (e) => {
        let ev; try { ev = JSON.parse(e.data); } catch (_) { return; }
        if (ev.job == null || ev.job !== myJobId) return;
        const st = el.querySelector('#mecha-status');
        const bar = el.querySelector('#mecha-bar');
        if (ev.type === 'progress' && bar) {
          bar.style.width = ev.total ? `${Math.round((ev.step / ev.total) * 100)}%` : '0%';
          if (st) st.textContent = `merging… ${ev.step}/${ev.total}`;
        } else if (ev.type === 'done') {
          if (st) st.textContent = `done — saved ${ev.output || ''}`;
          if (bar) bar.style.width = '100%';
          myJobId = null;
          loadFiles().then(() => renderMethod(el.querySelector('#mecha-method')?.value));
          setBusy(false);
        } else if (ev.type === 'error' || ev.type === 'cancelled') {
          if (st) st.textContent = ev.type === 'cancelled' ? 'cancelled' : `error: ${ev.message || 'merge failed'}`;
          myJobId = null;
          setBusy(false);
        }
      };

      function setBusy(b) {
        const btn = el.querySelector('#mecha-go');
        if (btn) { btn.disabled = b; btn.textContent = b ? 'Merging…' : 'Merge'; }
      }

      function fileOptions(selected) {
        const folder = el.querySelector('#mecha-folder')?.value || 'checkpoints';
        const list = FILES[folder] || [];
        return ['<option value="">— select —</option>']
          .concat(list.map((f) => `<option value="${esc(f)}"${f === selected ? ' selected' : ''}>${esc(f)}</option>`))
          .join('');
      }

      function modelSelect(label, hint) {
        const wrap = document.createElement('label');
        wrap.className = 'mecha-row';
        wrap.style.cssText = 'display:flex;align-items:center;gap:8px;margin:0 0 6px';
        wrap.innerHTML =
          `<span style="min-width:120px;font-size:12px;color:var(--txt-2)">${esc(label)}${hint ? ` <em style="color:var(--accent)">(${esc(hint)})</em>` : ''}</span>` +
          `<select class="mecha-model" style="flex:1">${fileOptions('')}</select>`;
        return wrap;
      }

      function renderMethod(id) {
        const slots = el.querySelector('#mecha-models');
        const ps = el.querySelector('#mecha-params');
        const doc = el.querySelector('#mecha-doc');
        if (!slots || !ps) return;
        slots.innerHTML = '';
        ps.innerHTML = '';
        const spec = METHODS[id];
        if (!spec) { doc.textContent = ''; return; }
        doc.textContent = spec.doc || '';

        (spec.models || []).forEach((m) => slots.appendChild(modelSelect(m.name, m.space)));

        // varargs methods (e.g. n_average) take an arbitrary number of models.
        if (spec.varargs) {
          const extra = document.createElement('div');
          extra.id = 'mecha-extra';
          const addBtn = document.createElement('button');
          addBtn.type = 'button';
          addBtn.className = 'btn small';
          addBtn.textContent = '+ add model';
          addBtn.onclick = () => extra.appendChild(modelSelect(`model ${extra.children.length + 1}`, ''));
          slots.appendChild(extra);
          slots.appendChild(addBtn);
          extra.appendChild(modelSelect('model 1', ''));
          extra.appendChild(modelSelect('model 2', ''));
        }

        (spec.params || []).forEach((p) => {
          const row = document.createElement('label');
          row.style.cssText = 'display:flex;align-items:center;gap:8px;margin:0 0 6px';
          let input;
          if (p.kind === 'bool') {
            input = `<input type="checkbox" class="mecha-param" data-name="${esc(p.name)}" data-kind="bool"${p.default ? ' checked' : ''}>`;
          } else {
            const type = (p.kind === 'int' || p.kind === 'float') ? 'number' : 'text';
            const step = p.kind === 'int' ? '1' : (p.kind === 'float' ? 'any' : '');
            const val = (p.default !== null && p.default !== undefined) ? esc(p.default) : '';
            input = `<input type="${type}"${step ? ` step="${step}"` : ''} class="mecha-param" data-name="${esc(p.name)}" data-kind="${esc(p.kind)}" value="${val}" placeholder="${esc(p.kind)}" style="flex:1">`;
          }
          row.innerHTML = `<span style="min-width:120px;font-size:12px;color:var(--txt-2)">${esc(p.name)}</span>${input}`;
          ps.appendChild(row);
        });
      }

      function renderForm(meta) {
        if (!meta.installed) {
          body.innerHTML = `
            <div class="meta" style="padding:14px;line-height:1.5">
              <b>sd-mecha is not installed.</b><br>
              Install it into Diffucore's environment, then <b>Reload</b> this extension
              from Settings → Extensions:
              <pre style="margin:8px 0 0">pip install sd-mecha</pre>
              ${meta.error ? `<div class="hint" style="margin-top:8px">${esc(meta.error)}</div>` : ''}
            </div>`;
          return;
        }
        const ids = Object.keys(METHODS).sort();
        body.innerHTML = `
          <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:0 0 10px">
            <label style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--txt-2)">folder
              <select id="mecha-folder">
                <option value="checkpoints">checkpoints</option>
                <option value="diffusion-models">diffusion-models</option>
              </select>
            </label>
            <label style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--txt-2)">method
              <input id="mecha-filter" type="text" placeholder="filter…" style="width:110px">
              <select id="mecha-method">${ids.map((i) => `<option value="${esc(i)}">${esc(i)}</option>`).join('')}</select>
            </label>
          </div>
          <pre class="hint" id="mecha-doc" style="white-space:pre-wrap;min-height:18px;margin:0 0 10px"></pre>

          <div style="font-size:12px;color:var(--txt-3);margin:0 0 4px">models</div>
          <div id="mecha-models" style="margin:0 0 12px"></div>

          <div id="mecha-params-wrap">
            <div style="font-size:12px;color:var(--txt-3);margin:0 0 4px">parameters</div>
            <div id="mecha-params" style="margin:0 0 12px"></div>
          </div>

          <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:0 0 10px">
            <label style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--txt-2)">output
              <input id="mecha-output" type="text" placeholder="merged-model" style="width:200px">
            </label>
            <label style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--txt-2)">device
              <select id="mecha-device"><option value="cpu">cpu</option><option value="cuda">cuda</option></select>
            </label>
            <label style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--txt-2)">dtype
              <select id="mecha-dtype"><option value="fp16">fp16</option><option value="fp32">fp32</option><option value="bf16">bf16</option></select>
            </label>
          </div>

          <div style="display:flex;align-items:center;gap:12px">
            <button class="btn primary" id="mecha-go">Merge</button>
            <span id="mecha-status" class="hint"></span>
          </div>
          <div style="height:4px;background:var(--surface-2);border-radius:2px;margin-top:10px;overflow:hidden">
            <div id="mecha-bar" style="height:100%;width:0;background:var(--accent);transition:width .2s"></div>
          </div>`;

        const methodSel = el.querySelector('#mecha-method');
        const filter = el.querySelector('#mecha-filter');
        const folder = el.querySelector('#mecha-folder');

        methodSel.addEventListener('change', () => renderMethod(methodSel.value));
        folder.addEventListener('change', () => renderMethod(methodSel.value)); // repopulate file options
        filter.addEventListener('input', () => {
          const q = filter.value.toLowerCase();
          const shown = ids.filter((i) => i.toLowerCase().includes(q));
          methodSel.innerHTML = shown.map((i) => `<option value="${esc(i)}">${esc(i)}</option>`).join('');
          renderMethod(methodSel.value);
        });

        el.querySelector('#mecha-go').addEventListener('click', async () => {
          const st = el.querySelector('#mecha-status');
          const method = methodSel.value;
          if (!method) { st.textContent = 'pick a method'; return; }
          const models = Array.from(el.querySelectorAll('.mecha-model'))
            .map((s) => s.value).filter(Boolean);
          if (!models.length) { st.textContent = 'select at least one model'; return; }
          const params = {};
          el.querySelectorAll('.mecha-param').forEach((inp) => {
            params[inp.dataset.name] = inp.dataset.kind === 'bool' ? inp.checked : inp.value;
          });
          const output = el.querySelector('#mecha-output').value.trim();
          if (!output) { st.textContent = 'name the output'; return; }

          setBusy(true);
          el.querySelector('#mecha-bar').style.width = '0%';
          st.textContent = 'queued…';
          try {
            const r = await fetch(`${API}/merge`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                method, folder: folder.value, models, params, output,
                device: el.querySelector('#mecha-device').value,
                dtype: el.querySelector('#mecha-dtype').value,
              }),
            });
            const d = await r.json();
            if (!r.ok) throw new Error(d.detail || 'merge failed');
            myJobId = d.job;
            st.textContent = 'queued — merging on the shared worker…';
          } catch (e) {
            st.textContent = `error: ${e.message || e}`;
            setBusy(false);
          }
        });

        renderMethod(methodSel.value);
      }

      Promise.all([loadFiles(), loadMethods()]).then(([, meta]) => renderForm(meta));
    },

    unmount(el) {
      if (el._mechaEs) { try { el._mechaEs.close(); } catch (_) {} el._mechaEs = null; }
    },
  });

  // ── settings panel (Settings → Extensions) ───────────────────────────────
  window.DiffucoreExt.registerSettingsPanel({
    id: NAME,
    title: 'Mecha Merge',
    mount(el) {
      el.innerHTML = `
        <h3 style="margin:0 0 6px;font-size:14px;color:var(--txt)">Mecha Merge</h3>
        <p class="sub" style="margin:0 0 8px;color:var(--txt-3);font-size:12px">
          Model merging via <a href="https://github.com/ljleb/sd-mecha" target="_blank" rel="noopener">sd-mecha</a>.
          The merge runs in the <b>Merge</b> tab.</p>
        <pre class="meta" id="mecha-set" style="min-height:40px">checking…</pre>`;
      const out = el.querySelector('#mecha-set');
      fetch(`${API}/methods`).then((r) => r.json()).then((d) => {
        if (!d.installed) {
          out.textContent = `sd-mecha not installed.\n${d.error || ''}\nInstall: pip install sd-mecha, then Reload.`;
        } else {
          out.textContent = `sd-mecha installed.\n${(d.methods || []).length} merge methods available.`;
        }
      }).catch((e) => { out.textContent = String(e); });
    },
  });
})();

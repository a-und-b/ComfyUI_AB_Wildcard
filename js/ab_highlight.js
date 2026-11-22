// Syntax highlighting for the AB Wildcard text widget.
//
// A mirror <div> sits behind the real <textarea>; the textarea keeps its text
// transparent and stays on top, so caret, selection, undo, paste and IME are the
// browser's own. The mirror repaints the same text with token colours and
// diagnostic underlines. Works in the legacy canvas and in Nodes 2.0: attach is
// idempotent and re-runs from a MutationObserver, because Nodes 2.0 re-creates
// the textarea (collapse/expand, tab switch).
import { tokenize, findMatches, replaceAll } from "./ab_syntax.js";

const CSS = `
.ab-hl-ta { color: transparent !important; background: transparent !important; position: relative; z-index: 1; caret-color: var(--ab-hl-text); }
.ab-hl-mirror { position: absolute; z-index: 0; margin: 0; overflow: hidden; pointer-events: none; color: var(--ab-hl-text); box-sizing: border-box; }
.ab-hl-status { position: absolute; z-index: 2; pointer-events: none; font: 11px/1.3 system-ui, sans-serif; padding: 2px 6px; border-radius: 4px; max-width: 95%; background: var(--ab-hl-status-bg); color: var(--ab-hl-text); box-shadow: 0 0 0 1px var(--ab-hl-status-line); }
.ab-hl-status[hidden] { display: none; }
.ab-hl-status b { font-weight: 600; }
.ab-hl-status .ab-error { color: var(--ab-hl-error); } .ab-hl-status .ab-warning { color: var(--ab-hl-warning); }
:root { --ab-hl-text: var(--input-text, #d8dee9);
  --ab-hl-comment: #7d8590; --ab-hl-var: #7ee2a8; --ab-hl-var-bg: rgba(46,160,100,.28); --ab-hl-assign: #8b949e;
  --ab-hl-wild: #5cc8ff; --ab-hl-seq: #3fe0c5; --ab-hl-tag: #c297ff; --ab-hl-lora: #ffb86c; --ab-hl-lora-bg: rgba(255,158,100,.18); --ab-hl-brace: #ffcb6b; --ab-hl-mode: #ff7ad9;
  --ab-hl-num: #ffa657; --ab-hl-kw: #ff8fa3;
  --ab-hl-error: #ff5c5c; --ab-hl-warning: #f0b429; --ab-hl-info: #5cc8ff;
  --ab-hl-status-bg: rgba(20,22,28,.92); --ab-hl-status-line: rgba(255,255,255,.18);
  --ab-hl-match: rgba(255,203,107,.30); --ab-hl-match-cur: rgba(255,150,40,.65); --ab-hl-match-line: #ffcb6b; }
:root.ab-hl-light { --ab-hl-text: var(--input-text, #24292f);
  --ab-hl-comment: #6e7781; --ab-hl-var: #116329; --ab-hl-var-bg: rgba(46,160,100,.18); --ab-hl-assign: #57606a;
  --ab-hl-wild: #0550ae; --ab-hl-seq: #0a7f6d; --ab-hl-tag: #6639ba; --ab-hl-lora: #9a4a00; --ab-hl-lora-bg: rgba(255,140,0,.14); --ab-hl-brace: #8a5a00; --ab-hl-mode: #b0198a;
  --ab-hl-num: #a04100; --ab-hl-kw: #b3263e;
  --ab-hl-error: #cf222e; --ab-hl-warning: #9a6700; --ab-hl-info: #0550ae;
  --ab-hl-status-bg: rgba(255,255,255,.95); --ab-hl-status-line: rgba(0,0,0,.2);
  --ab-hl-match: rgba(255,200,0,.35); --ab-hl-match-cur: rgba(255,140,0,.55); --ab-hl-match-line: #b36b00; }
.ab-t-comment { color: var(--ab-hl-comment); }
.ab-t-varDef, .ab-t-variable { color: var(--ab-hl-var); background: var(--ab-hl-var-bg); border-radius: 3px; box-shadow: 0 0 0 1px var(--ab-hl-var-bg); }
.ab-t-assign { color: var(--ab-hl-assign); }
.ab-t-wildcard { color: var(--ab-hl-wild); } .ab-t-wildcardSeq { color: var(--ab-hl-seq); }
.ab-t-angle, .ab-t-tagDelim, .ab-t-tagOp { color: var(--ab-hl-tag); }
.ab-t-lora { color: var(--ab-hl-lora); background: var(--ab-hl-lora-bg); border-radius: 3px; }
.ab-t-brace, .ab-t-pipe { color: var(--ab-hl-brace); }
.ab-t-seqMode { color: var(--ab-hl-mode); }
.ab-t-count, .ab-t-prob, .ab-t-weight { color: var(--ab-hl-num); }
.ab-t-ifKw, .ab-t-ifSep, .ab-t-fn { color: var(--ab-hl-kw); }
.ab-m { background: var(--ab-hl-match); border-radius: 2px; }
.ab-m-cur { background: var(--ab-hl-match-cur); box-shadow: 0 0 0 1px var(--ab-hl-match-line); }
.ab-find { position: absolute; z-index: 3; display: grid; gap: 3px; padding: 4px; border-radius: 6px; font: 12px/1.3 system-ui, sans-serif; background: var(--ab-hl-status-bg); color: var(--ab-hl-text); box-shadow: 0 0 0 1px var(--ab-hl-status-line), 0 2px 8px rgba(0,0,0,.3); }
.ab-find[hidden] { display: none; }
.ab-find div { display: flex; gap: 3px; align-items: center; }
.ab-find input { flex: 1; min-width: 0; width: 110px; font: inherit; padding: 2px 4px; color: var(--ab-hl-text); background: transparent; border: 1px solid var(--ab-hl-status-line); border-radius: 3px; }
.ab-find button { font: inherit; padding: 1px 6px; cursor: pointer; color: var(--ab-hl-text); background: transparent; border: 1px solid var(--ab-hl-status-line); border-radius: 3px; }
.ab-find button:hover { background: var(--ab-hl-match); }
.ab-find .ab-find-count { min-width: 40px; text-align: center; opacity: .8; }
.ab-find .ab-find-mode button[aria-pressed="true"] { background: var(--ab-hl-match); }
.ab-find.ab-find-wild { width: 300px; }
.ab-find.ab-find-wild .ab-find-count, .ab-find.ab-find-wild [data-a="prev"], .ab-find.ab-find-wild [data-a="next"], .ab-find.ab-find-wild .ab-find-rep { display: none; }
.ab-find .ab-find-list { display: block; max-height: 220px; overflow-y: auto; }
.ab-find .ab-find-list[hidden] { display: none; }
.ab-find .ab-find-row { cursor: pointer; padding: 2px 4px; border-radius: 3px; }
.ab-find .ab-find-row[aria-selected="true"] { background: var(--ab-hl-match); }
.ab-find .ab-find-tok { font-weight: 600; white-space: nowrap; }
.ab-find .ab-find-card { color: var(--ab-hl-tag); }
.ab-find .ab-find-prev { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; opacity: .75; }
.ab-find .ab-find-more { opacity: .7; white-space: nowrap; }
.ab-find .ab-find-empty { opacity: .7; padding: 2px 4px; }
.ab-d-error { text-decoration: underline wavy var(--ab-hl-error); text-underline-offset: 2px; }
.ab-d-warning { text-decoration: underline wavy var(--ab-hl-warning); text-underline-offset: 2px; }
.ab-d-info { text-decoration: underline dotted var(--ab-hl-info); text-underline-offset: 2px; }
`;

const COPIED = ["font", "lineHeight", "letterSpacing", "whiteSpace", "overflowWrap", "wordBreak", "tabSize", "textAlign", "borderRadius"];
const RANK = { error: 3, warning: 2, info: 1 };
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function isLight() {
  // ComfyUI marks the light theme on <html> (class or data attribute); fall back to the page background.
  const el = document.documentElement;
  if (/light/i.test(el.className) || /light/i.test(el.dataset.theme || "")) return true;
  const m = /rgba?\((\d+),\s*(\d+),\s*(\d+)/.exec(getComputedStyle(document.body).backgroundColor);
  return !!m && (+m[1] + +m[2] + +m[3]) / 3 > 140;
}

// Cut `text` at every token/diagnostic/match boundary and wrap each piece.
// matches: optional find ranges { start, end }; current: index of the strong one.
export function render(text, { tokens, diagnostics, matches = [], current = -1 }) {
  const cuts = new Set([0, text.length]);
  for (const r of tokens) { cuts.add(r.start); cuts.add(r.end); }
  for (const d of diagnostics) { cuts.add(Math.max(0, d.start)); cuts.add(Math.min(text.length, d.end)); }
  for (const m of matches) { cuts.add(m.start); cuts.add(m.end); }
  const points = [...cuts].sort((a, b) => a - b);
  let html = "", ti = 0;
  for (let p = 0; p < points.length - 1; p++) {
    const a = points[p], b = points[p + 1];
    if (b <= a) continue;
    while (ti < tokens.length && tokens[ti].end <= a) ti++;
    const tok = tokens[ti] && tokens[ti].start <= a ? tokens[ti] : null;
    let sev = null;
    for (const d of diagnostics) if (d.start <= a && d.end >= b && (!sev || RANK[d.severity] > RANK[sev])) sev = d.severity;
    const mi = matches.findIndex((m) => m.start <= a && m.end >= b);
    const cls = [tok && `ab-t-${tok.type}`, sev && `ab-d-${sev}`, mi >= 0 && (mi === current ? "ab-m ab-m-cur" : "ab-m")].filter(Boolean).join(" ");
    const piece = esc(text.slice(a, b));
    html += cls ? `<span class="${cls}">${piece}</span>` : piece;
  }
  return html + "\n"; // trailing newline keeps the last empty line scrollable like the textarea
}

export function installHighlighter({ resolveTextarea, getKnown, search }) {
  if (!document.getElementById("ab-hl-style")) {
    const st = document.createElement("style");
    st.id = "ab-hl-style"; st.textContent = CSS; document.head.appendChild(st);
  }
  const attached = new Set();
  let findMode = "text"; // the bar reopens in the mode it was last used in

  function paint(t) {
    const m = t.__abMirror;
    if (!m || !m.isConnected) return;
    const cs = getComputedStyle(t);
    for (const p of COPIED) m.style[p] = cs[p];
    m.style.left = t.offsetLeft + "px"; m.style.top = t.offsetTop + "px";
    m.style.width = t.offsetWidth + "px"; m.style.height = t.offsetHeight + "px";
    const sb = t.offsetWidth - t.clientWidth - parseFloat(cs.borderLeftWidth) - parseFloat(cs.borderRightWidth);
    m.style.padding = `${cs.paddingTop} calc(${cs.paddingRight} + ${sb}px) ${cs.paddingBottom} ${cs.paddingLeft}`;
    m.style.borderStyle = "solid"; m.style.borderColor = "transparent";
    m.style.borderWidth = `${cs.borderTopWidth} ${cs.borderRightWidth} ${cs.borderBottomWidth} ${cs.borderLeftWidth}`;
    const result = tokenize(t.value, getKnown());
    t.__abResult = result;
    const f = t.__abFind;
    if (f) {
      f.matches = findMatches(t.value, f.query);
      if (f.after !== undefined) {
        const next = f.matches.findIndex((x) => x.start >= f.after);
        f.current = next < 0 ? 0 : next; // wrap like stepFind
        f.after = undefined;
      }
      f.current = f.matches.length ? Math.min(Math.max(f.current, 0), f.matches.length - 1) : -1;
      t.__abBar.querySelector(".ab-find-count").textContent = f.query ? `${f.matches.length ? f.current + 1 : 0} / ${f.matches.length}` : "";
      t.__abBar.style.top = t.offsetTop + 4 + "px";
      t.__abBar.style.left = Math.max(t.offsetLeft, t.offsetLeft + t.offsetWidth - t.__abBar.offsetWidth - 20) + "px";
    }
    m.innerHTML = render(t.value, { ...result, ...f });
    if (f?.reveal) {
      f.reveal = false;
      const el = m.querySelector(".ab-m-cur");
      if (el && (el.offsetTop < t.scrollTop || el.offsetTop + el.offsetHeight > t.scrollTop + t.clientHeight)) {
        t.scrollTop = el.offsetTop - t.clientHeight / 2;
      }
    }
    m.scrollTop = t.scrollTop;
    m.scrollLeft = t.scrollLeft;
    status(t);
  }

  // ---- Find and replace (Cmd/Ctrl+F while the field has focus) ----
  function openFind(t) {
    if (!t.__abBar) {
      const b = document.createElement("div"); b.className = "ab-find";
      b.innerHTML = `<div class="ab-find-mode"><button data-a="mode-text" title="Find in the text">Text</button><button data-a="mode-wild" title="Search wildcard files and cards">Wildcards</button></div><div><input class="ab-find-q" placeholder="Find" aria-label="Find"><span class="ab-find-count"></span><button data-a="prev" title="Previous (Shift+Enter)">&#8593;</button><button data-a="next" title="Next (Enter)">&#8595;</button><button data-a="close" title="Close (Esc)">&#10005;</button></div><div class="ab-find-rep"><input class="ab-find-r" placeholder="Replace" aria-label="Replace"><button data-a="one">Replace</button><button data-a="all">All</button></div><div class="ab-find-list" role="listbox" hidden></div>`;
      t.parentElement.appendChild(b);
      t.__abBar = b;
      const q = b.querySelector(".ab-find-q"), r = b.querySelector(".ab-find-r");
      q.addEventListener("input", () => {
        if (findMode === "wild") { queueSearch(t); return; }
        t.__abFind.query = q.value; t.__abFind.current = 0; t.__abFind.reveal = true; schedule(t);
      });
      b.addEventListener("keydown", (e) => {
        if (e.key === "Escape") { e.preventDefault(); closeFind(t); }
        else if (findMode === "wild" && (e.key === "ArrowDown" || e.key === "ArrowUp")) { e.preventDefault(); moveSel(t, e.key === "ArrowDown" ? 1 : -1); }
        else if (findMode === "wild" && e.key === "Enter") { e.preventDefault(); pick(t, t.__abHits?.[t.__abSel]); }
        else if (e.key === "Enter") { e.preventDefault(); if (e.target === r) replaceOne(t, r.value); else stepFind(t, e.shiftKey ? -1 : 1); }
      });
      b.addEventListener("mousedown", (e) => { if (e.target.closest("button, .ab-find-row")) e.preventDefault(); });
      // Nodes 2.0 selects/drags the node on pointer events and can swallow the focus change.
      for (const ev of ["pointerdown", "mousedown", "click"]) b.addEventListener(ev, (e) => e.stopPropagation());
      b.addEventListener("click", (e) => {
        const row = e.target.closest(".ab-find-row");
        if (row) { pick(t, t.__abHits[+row.dataset.i]); return; }
        const a = e.target.closest("button")?.dataset.a;
        if (a === "mode-text" || a === "mode-wild") { setMode(t, a === "mode-wild" ? "wild" : "text"); return; }
        if (a === "prev") stepFind(t, -1); else if (a === "next") stepFind(t, 1);
        else if (a === "close") closeFind(t);
        else if (a === "one") replaceOne(t, r.value); else if (a === "all") replaceEvery(t, r.value);
      });
    }
    const sel = t.value.slice(t.selectionStart, t.selectionEnd);
    const q = t.__abBar.querySelector(".ab-find-q");
    if (sel && !sel.includes("\n")) q.value = sel;
    t.__abFind = { query: q.value, current: 0, matches: [], reveal: true };
    t.__abBar.hidden = false;
    setMode(t, findMode);
    q.focus(); q.select();
  }

  // ---- Wildcards mode: pick a wildcard file or card by name or content ----
  function setMode(t, mode) {
    findMode = mode;
    const b = t.__abBar, q = b.querySelector(".ab-find-q");
    b.classList.toggle("ab-find-wild", mode === "wild");
    for (const x of b.querySelectorAll(".ab-find-mode button")) x.setAttribute("aria-pressed", String(x.dataset.a === "mode-" + mode));
    q.placeholder = q.ariaLabel = mode === "wild" ? "Search wildcards" : "Find";
    b.querySelector(".ab-find-list").hidden = mode !== "wild";
    // Text highlights belong to Text mode only.
    t.__abFind.query = mode === "wild" ? "" : q.value; t.__abFind.current = 0; t.__abFind.reveal = true;
    if (mode === "wild") runSearch(t);
    schedule(t); // repaint: drops or restores the Text-mode marks
    q.focus();
  }

  function queueSearch(t) {
    clearTimeout(t.__abTimer);
    t.__abTimer = setTimeout(() => runSearch(t), 150);
  }

  async function runSearch(t) {
    const q = t.__abBar.querySelector(".ab-find-q").value.trim(), n = (t.__abSeq = (t.__abSeq || 0) + 1);
    let hits = [];
    if (q.length >= 2) { try { hits = await search(q); } catch { hits = []; } }
    if (n !== t.__abSeq || findMode !== "wild" || !t.__abFind) return; // stale answer
    t.__abHits = hits; t.__abSel = 0;
    const list = t.__abBar.querySelector(".ab-find-list");
    list.innerHTML = hits.length
      ? hits.map((h, i) => `<div class="ab-find-row" role="option" data-i="${i}"><span class="ab-find-tok${h.kind === "card" ? " ab-find-card" : ""}">${esc(h.token)}</span><span class="ab-find-prev">${esc(h.preview)}</span><span class="ab-find-more">+${h.more}</span></div>`).join("")
      : `<div class="ab-find-empty">${q.length < 2 ? "Type at least 2 characters" : "No wildcards found"}</div>`;
    markSel(t);
  }

  function markSel(t) {
    t.__abBar.querySelectorAll(".ab-find-row").forEach((r, i) => {
      r.setAttribute("aria-selected", String(i === t.__abSel));
      if (i === t.__abSel) r.scrollIntoView({ block: "nearest" });
    });
  }

  function moveSel(t, dir) {
    const n = t.__abHits?.length;
    if (!n) return;
    t.__abSel = (t.__abSel + dir + n) % n;
    markSel(t);
  }

  function pick(t, hit) {
    if (!hit) return;
    // The textarea keeps its selection while focus is in the bar.
    insert(t, t.selectionStart, t.selectionEnd, `__${hit.token}__`);
    closeFind(t);
  }

  function closeFind(t) {
    if (!t.__abFind) return;
    clearTimeout(t.__abTimer); t.__abSeq = (t.__abSeq || 0) + 1;
    t.__abFind = undefined;
    t.__abBar.hidden = true;
    t.focus();
    schedule(t);
  }

  function stepFind(t, dir) {
    const f = t.__abFind, n = f?.matches.length;
    if (!n) return;
    f.current = (f.current + dir + n) % n; f.reveal = true;
    schedule(t);
  }

  // Through execCommand so native undo survives and the field fires "input" (widget sync).
  function insert(t, start, end, text) {
    t.focus({ preventScroll: true });
    t.setSelectionRange(start, end);
    document.execCommand("insertText", false, text);
  }

  function replaceOne(t, text) {
    const f = t.__abFind, m = f?.matches[f.current];
    if (!m) return;
    // The paint below picks the first match after the inserted text, so a replacement
    // that contains the query ("a" -> "aa") is not hit again.
    f.after = m.start + text.length;
    insert(t, m.start, m.end, text);
    f.reveal = true;
    // Repaint now, not on the deferred frame: a fast second Enter (or a hidden
    // tab, where timers wait ~1s) would otherwise replace from stale matches.
    paint(t);
    t.__abBar.querySelector(".ab-find-r").focus();
  }

  function replaceEvery(t, text) {
    const f = t.__abFind;
    if (!f?.matches.length) return;
    insert(t, 0, t.value.length, replaceAll(t.value, f.query, text));
    paint(t);
    t.__abBar.querySelector(".ab-find-r").focus();
  }

  document.addEventListener("keydown", (e) => {
    const t = e.target;
    if ((e.metaKey || e.ctrlKey) && !e.altKey && e.key.toLowerCase() === "f") {
      const bar = t.closest?.(".ab-find");
      if (t.__abMirror) openFind(t);
      else if (bar) bar.querySelector(".ab-find-q").select();
      else return;
      e.preventDefault(); e.stopPropagation();
    } else if (e.key === "Escape" && t.__abFind) closeFind(t);
  }, true);

  function status(t) {
    const s = t.__abStatus;
    if (!s) return;
    const ds = t.__abResult.diagnostics.filter((d) => d.severity !== "info");
    const caret = t.selectionStart ?? 0;
    const here = ds.filter((d) => d.start <= caret && caret <= d.end).sort((a, b) => RANK[b.severity] - RANK[a.severity])[0];
    const errors = ds.filter((d) => d.severity === "error").length, warns = ds.length - errors;
    if (!ds.length) { s.hidden = true; return; }
    const counts = [errors && `<span class="ab-error">${errors} error${errors > 1 ? "s" : ""}</span>`, warns && `<span class="ab-warning">${warns} warning${warns > 1 ? "s" : ""}</span>`].filter(Boolean).join(" · ");
    const first = here || ds.sort((a, b) => RANK[b.severity] - RANK[a.severity] || a.start - b.start)[0];
    s.innerHTML = `${counts} — <b class="ab-${first.severity}">${esc(first.message)}</b>`;
    s.style.left = t.offsetLeft + 4 + "px";
    s.style.top = t.offsetTop + t.offsetHeight - s.offsetHeight - 4 + "px";
    s.hidden = false;
    s.style.top = t.offsetTop + t.offsetHeight - s.offsetHeight - 4 + "px";
  }

  // Coalesce into one animation frame; the timeout covers a hidden page, where
  // requestAnimationFrame is paused.
  function defer(fn) {
    let done = false;
    const run = () => { if (!done) { done = true; fn(); } };
    requestAnimationFrame(run); setTimeout(run, 60);
  }

  function schedule(t) {
    if (t.__abFrame) return;
    t.__abFrame = 1;
    defer(() => { t.__abFrame = 0; paint(t); });
  }

  function attach(t) {
    if (t.__abMirror?.isConnected) return;
    t.__abMirror?.remove(); t.__abStatus?.remove();
    document.documentElement.classList.toggle("ab-hl-light", isLight());
    const m = document.createElement("div"); m.className = "ab-hl-mirror";
    const s = document.createElement("div"); s.className = "ab-hl-status"; s.hidden = true;
    t.parentElement.insertBefore(m, t);
    t.parentElement.appendChild(s);
    // The textarea goes transparent, so the mirror carries its background. Opaque,
    // it also hides the wrapper's hover/focus tint (Nodes 2.0: hover:bg-muted).
    // ponytail: read once; a theme switch keeps the old colour until reload.
    t.__abBg ??= getComputedStyle(t).backgroundColor;
    m.style.background = t.__abBg;
    t.classList.add("ab-hl-ta");
    t.__abMirror = m; t.__abStatus = s;
    if (!attached.has(t)) {
      attached.add(t);
      const f = () => schedule(t);
      t.addEventListener("input", f);
      t.addEventListener("scroll", f);
      for (const ev of ["keyup", "click", "focus"]) t.addEventListener(ev, () => status(t));
      new ResizeObserver(f).observe(t);
      // `el.value = x` (workflow load, pickers, Vue's v-model) fires no input event.
      const d = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value");
      Object.defineProperty(t, "value", { configurable: true, get() { return d.get.call(this); }, set(v) { d.set.call(this, v); f(); } });
    }
    schedule(t);
  }

  function scan() {
    for (const t of document.querySelectorAll("textarea")) if (resolveTextarea(t)) attach(t);
  }
  let scanning = 0;
  const queueScan = () => { if (!scanning) { scanning = 1; defer(() => { scanning = 0; scan(); }); } };
  new MutationObserver(queueScan).observe(document.body, { childList: true, subtree: true });
  document.addEventListener("focusin", queueScan);
  scan();

  return {
    // Known names changed (refresh button, first fetch): repaint every attached textarea.
    repaintAll() {
      document.documentElement.classList.toggle("ab-hl-light", isLight());
      for (const t of document.querySelectorAll("textarea")) if (t.__abMirror) schedule(t);
    },
  };
}

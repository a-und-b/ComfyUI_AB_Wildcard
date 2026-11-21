// Syntax highlighting for the AB Wildcard text widget.
//
// A mirror <div> sits behind the real <textarea>; the textarea keeps its text
// transparent and stays on top, so caret, selection, undo, paste and IME are the
// browser's own. The mirror repaints the same text with token colours and
// diagnostic underlines. Works in the legacy canvas and in Nodes 2.0: attach is
// idempotent and re-runs from a MutationObserver, because Nodes 2.0 re-creates
// the textarea (collapse/expand, tab switch).
import { tokenize } from "./ab_syntax.js";

const CSS = `
.ab-hl-ta { color: transparent !important; background: transparent !important; position: relative; z-index: 1; caret-color: var(--ab-hl-text); }
.ab-hl-mirror { position: absolute; z-index: 0; margin: 0; overflow: hidden; pointer-events: none; color: var(--ab-hl-text); box-sizing: border-box; }
.ab-hl-status { position: absolute; z-index: 2; pointer-events: none; font: 11px/1.3 system-ui, sans-serif; padding: 2px 6px; border-radius: 4px; max-width: 95%; background: var(--ab-hl-status-bg); color: var(--ab-hl-text); box-shadow: 0 0 0 1px var(--ab-hl-status-line); }
.ab-hl-status[hidden] { display: none; }
.ab-hl-status b { font-weight: 600; }
.ab-hl-status .ab-error { color: var(--ab-hl-error); } .ab-hl-status .ab-warning { color: var(--ab-hl-warning); }
:root { --ab-hl-text: var(--input-text, #d8dee9);
  --ab-hl-comment: #7d8590; --ab-hl-var: #7ee2a8; --ab-hl-var-bg: rgba(46,160,100,.28); --ab-hl-assign: #8b949e;
  --ab-hl-wild: #5cc8ff; --ab-hl-seq: #3fe0c5; --ab-hl-tag: #c297ff; --ab-hl-brace: #ffcb6b; --ab-hl-mode: #ff7ad9;
  --ab-hl-num: #ffa657; --ab-hl-kw: #ff8fa3;
  --ab-hl-error: #ff5c5c; --ab-hl-warning: #f0b429; --ab-hl-info: #5cc8ff;
  --ab-hl-status-bg: rgba(20,22,28,.92); --ab-hl-status-line: rgba(255,255,255,.18);
  --ab-hl-match: rgba(255,203,107,.30); --ab-hl-match-cur: rgba(255,150,40,.65); --ab-hl-match-line: #ffcb6b; }
:root.ab-hl-light { --ab-hl-text: var(--input-text, #24292f);
  --ab-hl-comment: #6e7781; --ab-hl-var: #116329; --ab-hl-var-bg: rgba(46,160,100,.18); --ab-hl-assign: #57606a;
  --ab-hl-wild: #0550ae; --ab-hl-seq: #0a7f6d; --ab-hl-tag: #6639ba; --ab-hl-brace: #8a5a00; --ab-hl-mode: #b0198a;
  --ab-hl-num: #a04100; --ab-hl-kw: #b3263e;
  --ab-hl-error: #cf222e; --ab-hl-warning: #9a6700; --ab-hl-info: #0550ae;
  --ab-hl-status-bg: rgba(255,255,255,.95); --ab-hl-status-line: rgba(0,0,0,.2);
  --ab-hl-match: rgba(255,200,0,.35); --ab-hl-match-cur: rgba(255,140,0,.55); --ab-hl-match-line: #b36b00; }
.ab-t-comment { color: var(--ab-hl-comment); }
.ab-t-varDef, .ab-t-variable { color: var(--ab-hl-var); background: var(--ab-hl-var-bg); border-radius: 3px; box-shadow: 0 0 0 1px var(--ab-hl-var-bg); }
.ab-t-assign { color: var(--ab-hl-assign); }
.ab-t-wildcard { color: var(--ab-hl-wild); } .ab-t-wildcardSeq { color: var(--ab-hl-seq); }
.ab-t-angle, .ab-t-tagDelim, .ab-t-tagOp { color: var(--ab-hl-tag); }
.ab-t-brace, .ab-t-pipe { color: var(--ab-hl-brace); }
.ab-t-seqMode { color: var(--ab-hl-mode); }
.ab-t-count, .ab-t-prob, .ab-t-weight { color: var(--ab-hl-num); }
.ab-t-ifKw, .ab-t-ifSep, .ab-t-fn { color: var(--ab-hl-kw); }
.ab-m { background: var(--ab-hl-match); border-radius: 2px; }
.ab-m-cur { background: var(--ab-hl-match-cur); box-shadow: 0 0 0 1px var(--ab-hl-match-line); }
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

export function installHighlighter({ resolveTextarea, getKnown }) {
  if (!document.getElementById("ab-hl-style")) {
    const st = document.createElement("style");
    st.id = "ab-hl-style"; st.textContent = CSS; document.head.appendChild(st);
  }
  const attached = new Set();

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
    m.innerHTML = render(t.value, { ...result, ...t.__abFind });
    m.scrollTop = t.scrollTop;
    m.scrollLeft = t.scrollLeft;
    status(t);
  }

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

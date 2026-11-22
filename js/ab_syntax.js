// Tokenizer for the AB Wildcard prompt syntax. Pure and DOM-free so it can be
// tested with plain node (tests/syntax.test.mjs). Source of truth for the
// grammar is nodes.py, not the README.
//
// tokenize(text) -> { tokens, diagnostics }
//   tokens:       flat, ordered, non-overlapping { type, start, end } over `text`
//                 (offsets into the original string); gaps are plain text.
//   diagnostics:  { severity: "error", start, end, message }
//
// Token types: comment, varDef, assign, variable, wildcard, wildcardSeq,
//   angle, tagDelim, tagOp, brace, seqMode, count, prob, weight, pipe,
//   ifKw, ifSep, fn.

const NAME = "[A-Za-z0-9_]+";
const VAR_RE = new RegExp(`\\$(${NAME})((?:\\.[A-Za-z_]+)*)`, "y");
const ASSIGN_RE = new RegExp(`\\$(${NAME})(\\s*)=`, "y");
const COUNT_RE = /\s*\d+(?:\s*-\s*\d+)?\s*\$\$(?:[^|{}$]*\$\$)?/y;
const PROB_RE = /\d+(?:\.\d+)?%/y;
const WEIGHT_RE = /\d+(?:\.\d+)?::/y;
const IF_RE = /\[if\s/iy;
const FN_RE = /\[(?:clean|shuffle):/iy;
const GEN_RE = /\[gen(?:\s+(\w+))?\s*(?:\(([^():\]]*)\))?\s*:/iy; // ops.py _MARKER
const GEN_OPS = ["fill", "spin", "expand", "short", "fit", "do", "theme", "angle", "avoid"]; // ops.py LLM_OPS + LLM_DIRECTIONS
const TAG_OP_RE = /\b(?:AND|OR|NOT|XOR)\b|--|[()]/iy;

// Where the engine cuts a line: first "//", then first "#", with "__#" and
// "<#" protected (nodes.py, ABWildcardNode.process).
function commentStart(line) {
  let cut = line.length;
  const slash = line.indexOf("//");
  if (slash >= 0) cut = slash;
  for (let i = line.indexOf("#"); i >= 0 && i < cut; i = line.indexOf("#", i + 1)) {
    const protectedHash = line.slice(Math.max(0, i - 2), i) === "__" || line[i - 1] === "<";
    if (!protectedHash) { cut = i; break; }
  }
  return cut;
}

function sticky(re, text, at) {
  re.lastIndex = at;
  return re.exec(text);
}

// known (optional, from /ab_wildcard/wildcards): { wildcards, tags, globals } as
// arrays or Sets of names. Without it the "unknown name" and "unset variable"
// rules stay silent about what only the server can know.
export function tokenize(text, known = null) {
  const tokens = [];
  const diagnostics = [];
  const add = (type, start, end) => { if (end > start) tokens.push({ type, start, end }); };
  const diag = (severity, start, end, message) => diagnostics.push({ severity, start, end, message });
  const error = (start, end, message) => diag("error", start, end, message);
  const warn = (start, end, message) => diag("warning", start, end, message);
  const seenDecks = new Map(); // normalized {!…} content -> true

  // Per-line comment ranges and line-start positions.
  const comments = new Map(); // comment start -> end
  const lineStarts = new Set([0]);
  for (let pos = 0; pos <= text.length;) {
    let nl = text.indexOf("\n", pos);
    if (nl < 0) nl = text.length;
    const cut = pos + commentStart(text.slice(pos, nl));
    if (cut < nl) comments.set(cut, nl);
    lineStarts.add(nl + 1);
    pos = nl + 1;
  }

  // Frames: { kind: "brace" | "if" | "fn", start, head? }
  const stack = [];
  let lineHasText = false; // false while only whitespace seen on this line

  // Option-start markers (after "{" or "|"): weight "3::", and at block start
  // also mode ~@!, count "N$$" or "N-M$$ sep $$", probability "30%".
  function optionStart(i, atBlockStart) {
    let at = i;
    if (atBlockStart) {
      const frame = stack[stack.length - 1];
      if ("~@!".includes(text[at] || "\0")) {
        frame.mode = text[at];
        if (frame.mode === "@" && stack.slice(0, -1).some((f) => f.kind === "brace" && f.mode === "@")) {
          warn(frame.start, at + 1, "{@…} inside another {@…} is not counted evenly");
        }
        add("seqMode", at, at + 1); at += 1;
      }
      let m = sticky(COUNT_RE, text, at);
      if (m) {
        frame.low = parseInt(m[0], 10);
        add("count", at, at + m[0].length); return at + m[0].length;
      }
      m = sticky(PROB_RE, text, at);
      if (m) { add("prob", at, at + m[0].length); return at + m[0].length; }
    }
    const w = sticky(WEIGHT_RE, text, at);
    if (w) { add("weight", at, at + w[0].length); at += w[0].length; }
    return at;
  }

  // Scan the inside of "<...>" up to `end` (exclusive of the final ">").
  function scanAngle(open, close) {
    const inner = text.slice(open + 1, close);
    if (!inner.includes("[")) { add("angle", open, close + 1); return; }
    add("tagDelim", open, open + 1);
    for (let i = open + 1; i < close;) {
      const ch = text[i];
      if (ch === "[" || ch === "]") { add("tagDelim", i, i + 1); i++; continue; }
      if (ch === "$") {
        const m = sticky(VAR_RE, text, i);
        if (m && i + m[0].length <= close) { add("variable", i, i + m[0].length); i += m[0].length; continue; }
      }
      const op = sticky(TAG_OP_RE, text, i);
      if (op && (i === open + 1 || !/\w/.test(text[i - 1]) || op[0] === "--" || /[()]/.test(op[0]))) {
        add("tagOp", i, i + op[0].length); i += op[0].length; continue;
      }
      i++;
    }
    add("tagDelim", close, close + 1);
  }

  // Rules that need the whole block: empty, count larger than the options,
  // two identical {!…} decks (they draw the same option).
  function checkBlock(block, close) {
    const body = text.slice(block.start + 1, close);
    if (!body.trim()) { warn(block.start, close + 1, "Empty {} has no options"); return; }
    if (block.low !== null && block.low > block.options) {
      warn(block.start, close + 1, `Count ${block.low} is larger than the ${block.options} option(s)`);
    }
    if (block.mode === "!") {
      const key = body.replace(/\s+/g, " ").trim();
      if (seenDecks.has(key)) warn(block.start, close + 1, "Identical {!…} blocks draw the same option");
      seenDecks.set(key, true);
    }
  }

  let i = 0;
  while (i < text.length) {
    if (comments.has(i)) {
      const end = comments.get(i);
      const before = text.slice(0, i);
      if (stack.length) diag("info", i, end, "Inside a block this starts a comment; the rest of the line is ignored");
      else if (/https?:$/i.test(before)) diag("info", i - 1, end, "'//' starts a comment, even in a URL");
      add("comment", i, end); i = end; continue;
    }
    const ch = text[i];

    if (ch === "\n") { lineHasText = false; i++; continue; }
    if (/\s/.test(ch)) { i++; continue; }

    // Assignment at line start: "$name = value" (value is scanned on).
    if (ch === "$" && !lineHasText) {
      const m = sticky(ASSIGN_RE, text, i);
      if (m) {
        const nameEnd = i + 1 + m[1].length;
        add("varDef", i, nameEnd);
        add("assign", nameEnd + m[2].length, nameEnd + m[2].length + 1);
        i = nameEnd + m[2].length + 1;
        lineHasText = true;
        continue;
      }
    }
    lineHasText = true;

    if (ch === "$") {
      if (text[i + 1] === "$") { i += 2; continue; } // "$$" of a count, not a variable
      const m = sticky(VAR_RE, text, i);
      if (m) { add("variable", i, i + m[0].length); i += m[0].length; continue; }
      i++; continue;
    }

    if (ch === "_" && text[i + 1] === "_") {
      // __name__ ends at the next "__" or ">" (engine regex), same line only.
      const nl = text.indexOf("\n", i);
      const lineEnd = nl < 0 ? text.length : nl;
      const rest = text.slice(i + 2, lineEnd);
      const m = /__|>/.exec(rest);
      if (m && m.index > 0) {
        const end = i + 2 + m.index + m[0].length;
        const seq = /^(?:\d+(?:\s*-\s*\d+)?\$\$)?[~@!]|^[~@!]/.test(rest);
        add(seq ? "wildcardSeq" : "wildcard", i, end);
        i = end; continue;
      }
      i += 2; continue;
    }

    if (ch === "<") {
      const nl = text.indexOf("\n", i);
      const lineEnd = nl < 0 ? text.length : nl;
      const close = text.indexOf(">", i + 1);
      const inner = close >= 0 && close < lineEnd ? text.slice(i + 1, close) : null;
      if (inner !== null && inner.length && (inner.includes("[") || (inner === inner.trim() && !inner.includes("__")))) {
        scanAngle(i, close);
        i = close + 1; continue;
      }
      if (text[i + 1] === "[") error(i, i + 2, "Unclosed <[ … ]>");
      i++; continue;
    }

    if (ch === "{") {
      stack.push({ kind: "brace", start: i, options: 1, mode: null, low: null });
      add("brace", i, i + 1);
      i = optionStart(i + 1, true);
      continue;
    }
    if (ch === "}") {
      let k = stack.length - 1;
      while (k >= 0 && stack[k].kind !== "brace") k--;
      if (k < 0) { error(i, i + 1, "Unmatched }"); i++; continue; }
      for (let j = stack.length - 1; j > k; j--) error(stack[j].start, stack[j].start + 1, unclosedMsg(stack[j]));
      const block = stack[k];
      stack.length = k;
      checkBlock(block, i);
      add("brace", i, i + 1);
      i++; continue;
    }

    if (ch === "|") {
      const top = stack[stack.length - 1];
      if (top && top.kind === "brace") top.options++;
      if (top && top.kind !== "fn") {
        add("pipe", i, i + 1);
        if (top.kind === "if") top.head = false;
        i = top.kind === "brace" ? optionStart(i + 1, false) : i + 1;
        continue;
      }
      i++; continue;
    }

    if (ch === "[") {
      let m = sticky(IF_RE, text, i);
      if (m) {
        stack.push({ kind: "if", start: i, head: true });
        add("ifKw", i, i + 3);
        i += 3; continue;
      }
      m = sticky(FN_RE, text, i);
      if (m) {
        stack.push({ kind: "fn", start: i });
        add("fn", i, i + m[0].length);
        i += m[0].length; continue;
      }
      m = sticky(GEN_RE, text, i);
      if (m) {
        stack.push({ kind: "fn", start: i, gen: true });
        add("fn", i, i + m[0].length);
        if (m[1] && !GEN_OPS.includes(m[1].toLowerCase())) {
          warn(i, i + m[0].length, `Unknown [gen …] operation "${m[1]}": use ${GEN_OPS.join(", ")}`);
        } else if (m[1] && m[1].toLowerCase() === "do" && !(m[2] || "").trim()) {
          warn(i, i + m[0].length, "[gen do …] needs an instruction: [gen do (what to do): …]");
        }
        i += m[0].length; continue;
      }
      i++; continue;
    }

    if (ch === ":") {
      const top = stack[stack.length - 1];
      if (top && top.kind === "if" && top.head) { top.head = false; top.colon = true; add("ifSep", i, i + 1); }
      i++; continue;
    }

    if (ch === "]") {
      let k = stack.length - 1;
      while (k >= 0 && stack[k].kind === "brace") k--;
      if (k < 0) { i++; continue; } // a plain "]" in prose
      for (let j = stack.length - 1; j > k; j--) error(stack[j].start, stack[j].start + 1, unclosedMsg(stack[j]));
      const frame = stack[k];
      if (frame.kind === "if" && !frame.colon) error(frame.start, frame.start + 3, "[if needs a ':' after the condition");
      add(frame.kind === "if" ? "ifKw" : "fn", i, i + 1);
      stack.length = k;
      i++; continue;
    }

    i++;
  }

  for (const f of stack) error(f.start, f.start + 1, unclosedMsg(f));
  tokens.sort((a, b) => a.start - b.start);
  checkNames(text, tokens, known, warn);
  return { tokens, diagnostics };
}

const lower = (s) => s.toLowerCase();
const asSet = (a) => new Set([...(a || [])].map(lower));

// Warnings that need names: variables never assigned, wildcards and tags the
// server does not know. Silent for what `known` does not provide.
function checkNames(text, tokens, known, warn) {
  const wildcards = known?.wildcards ? asSet(known.wildcards) : null;
  const tags = known?.tags ? asSet(known.tags) : null;
  const defined = asSet(known?.globals);
  let usesWildcard = false;

  for (const t of tokens) {
    if (t.type === "varDef") defined.add(lower(text.slice(t.start + 1, t.end)));
    if (t.type === "wildcard" || t.type === "wildcardSeq" || t.type === "angle" || t.type === "tagDelim") usesWildcard = true;
  }

  for (const t of tokens) {
    if (t.type === "variable") {
      const name = /^\$([A-Za-z0-9_]+)/.exec(text.slice(t.start, t.end))[1];
      // CSV wildcards and cards can define variables the frontend cannot see;
      // a leading digit is a price like "$5", not a variable.
      if (!usesWildcard && /^[A-Za-z_]/.test(name) && !defined.has(lower(name))) {
        warn(t.start, t.end, `$${name} is never assigned`);
      }
    } else if (wildcards && (t.type === "wildcard" || t.type === "wildcardSeq")) {
      let name = text.slice(t.start, t.end).replace(/^__/, "").replace(/(__|>)$/, "");
      while (/^(?:#[0-9|]+\$\$|[~@!]|\d+(?:\s*-\s*\d+)?\$\$)/.test(name)) name = name.replace(/^(?:#[0-9|]+\$\$|[~@!]|\d+(?:\s*-\s*\d+)?\$\$)/, "");
      const key = lower(name.trim());
      if (key && !/[*?]/.test(key) && !wildcards.has(key) && ![...wildcards].some((w) => w.endsWith("/" + key))) {
        warn(t.start, t.end, `Unknown wildcard "${name.trim()}"`);
      }
    }
  }

  // Tag names inside <[ … ]>: text between operators and variables.
  if (tags) {
    for (let k = 0; k < tokens.length; k++) {
      if (tokens[k].type !== "tagDelim" || text[tokens[k].start] !== "[") continue;
      let end = k + 1;
      while (end < tokens.length && !(tokens[end].type === "tagDelim" && text[tokens[end].start] === "]")) end++;
      if (end >= tokens.length) continue;
      let from = tokens[k].end;
      const flush = (to) => {
        const raw = text.slice(from, to);
        const name = raw.trim();
        if (name && !tags.has(lower(name))) warn(from + raw.indexOf(name), from + raw.indexOf(name) + name.length, `Unknown tag "${name}"`);
      };
      for (let m = k + 1; m < end; m++) { flush(tokens[m].start); from = tokens[m].end; }
      flush(tokens[end].start);
    }
  }
}

function unclosedMsg(frame) {
  return frame.kind === "brace" ? "Unclosed {" : frame.kind === "if" ? "Unclosed [if"
    : frame.gen ? "Unclosed [gen" : "Unclosed [clean: / [shuffle:";
}

// ---- Autocomplete ----------------------------------------------------------

export const MODIFIERS = ["clean", "upper", "lower", "title"];

// What is the user typing at the end of `before` (the text up to the cursor)?
// -> { kind: "modifier" | "variable" | "tag" | "wildcard", query, start, opener } or null.
// opener is "." for modifiers, "$" for variables, "<[" for tags, "__" or "<" for wildcards.
// Modifiers open from the first letter after the dot (a sentence-ending "$a." stays quiet).
// `{2$$` and "$5" are not variables; "a < b" is not a wildcard.
export function completionContext(before) {
  let m = /\$[A-Za-z0-9_]+(?:\.[A-Za-z_]+)*(\.)([A-Za-z_]+)$/.exec(before);
  if (m) return { kind: "modifier", query: m[2], start: m.index + m[0].length - m[2].length - 1, opener: "." };
  m = /(?<!\$)\$([A-Za-z_][A-Za-z0-9_]*)?$/.exec(before);
  if (m) return { kind: "variable", query: m[1] || "", start: m.index, opener: "$" };
  m = /(__|<\[)([\w\/\-\s]*)$/.exec(before);
  if (m) return { kind: m[1] === "<[" ? "tag" : "wildcard", query: m[2], start: m.index, opener: m[1] };
  m = /<(\w[\w\/\-]*)$/.exec(before);
  if (m) return { kind: "wildcard", query: m[1], start: m.index, opener: "<" };
  return null;
}

// The text that replaces [start, cursor) when `name` is accepted.
export function completionText(ctx, name) {
  if (ctx.kind === "tag") return `<[${name}]>`;
  if (ctx.kind === "wildcard") return ctx.opener === "<" ? `<${name}>` : `__${name}__`;
  return ctx.opener + name; // "$name" or ".modifier"
}

// Variable names for the popup: globals plus every `$name =` in the field,
// whole text (the engine collects all assignments before it substitutes), once
// per name, case-insensitive.
export function variableNames(text, globals = []) {
  const seen = new Set();
  const out = [];
  const add = (n) => { if (!seen.has(lower(n))) { seen.add(lower(n)); out.push(n); } };
  for (const g of globals || []) add(g);
  for (const t of tokenize(text).tokens) if (t.type === "varDef") add(text.slice(t.start + 1, t.end));
  return out;
}

// ---- Find and replace ------------------------------------------------------

// Case-insensitive, literal (no regex, no whole-word). -> [{ start, end }]
const literalRe = (query) => new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");

export function findMatches(text, query) {
  if (!query) return [];
  return [...text.matchAll(literalRe(query))].map((m) => ({ start: m.index, end: m.index + m[0].length }));
}

// Every match replaced; `with_` is literal too ("$&" and friends mean nothing).
export function replaceAll(text, query, with_) {
  if (!query) return text;
  return text.replace(literalRe(query), () => with_);
}

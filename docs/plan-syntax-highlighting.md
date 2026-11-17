# Plan: syntax highlighting and diagnostics for the text widget

Status: planned, not started. Written against 1.2.0 (sequences committed).
Format note: this repo has no `.ab/` yet; the plan lives here so any session can read it.

## Goal

While typing in the AB Wildcard node's `text` widget, wildcard syntax is
coloured and mistakes are marked, in both ComfyUI renderers (legacy canvas and
Nodes 2.0). Observable outcome: type `$c = <[color]>` and `{!a|b` and see the
tokens coloured and the unclosed `{` underlined with a message, with undo,
paste, autocomplete and the serialized value behaving exactly as before.

## Decisions (agreed with Holger)

1. **Own highlight overlay, no library.** The real `<textarea>` stays on top and
   transparent; a mirror `<div>` behind it paints the tokens. Native undo, paste
   and IME keep working. Research: the frontend bundles no reusable editor
   (only Tiptap, hash-named and tied to the agent panel); CodeMirror 6 would need
   a build step and replace the textarea; OverType, highlight-within-textarea,
   code-input are unverified against ComfyUI.
2. **Chips are styled spans**, not atomic widgets. A variable token gets a
   background and colour. Padding, icons and click targets are out, because the
   mirror text must keep exactly the textarea's glyph widths.
3. **No idle "rendered" view.** The Markdown-widget pattern (rendered while
   unfocused, textarea on double click) was considered and dropped.
4. **Diagnostics are part of this plan**, shown as underlines plus a message on
   hover or in a small strip under the field.
5. **One tokenizer, pure and DOM-free**, so it can be tested without a browser.

## Not in scope

- Atomic chips, editor replacement, CodeMirror, Monaco, Tiptap.
- A rendered idle view; syntax-aware editing (auto-pairing, folding).
- Changing the Python engine or its grammar. Where the engine is lenient, the
  highlighter follows it and, at most, warns.
- `$` and `<name>` autocomplete. Likely a follow-up that reuses the tokenizer
  (variables defined in the text are found by scanning); noted, not planned here.
- Exposing `globals.yaml` variable names to the frontend (needs an endpoint
  change). Until then a variable that only `globals.yaml` defines may warn as
  unset; see Open questions.

## Architecture

- `js/ab_syntax.js`: `tokenize(text) -> { tokens, diagnostics }`, no DOM, no ComfyUI
  imports. Imported by `js/umi_wildcards.js`. Test files must not live in `js/`:
  `WEB_DIRECTORY = "./js"` serves every file there to the browser.
- `js/umi_wildcards.js`: the overlay layer, attached through the existing
  document-level delegation. `resolveTextarea()` already resolves the textarea in
  both renderers; add a `focusin`/`input` hook plus a `MutationObserver`, because
  in Nodes 2.0 the Vue textarea can be re-created (the legacy `widget.element` is
  an orphan there).
- Overlay mechanics to settle in the spike: wrapper placement inside the
  textarea's parent so it follows zoom and pan in the legacy canvas; copying
  font, line-height, padding, white-space and wrap from computed style;
  scroll and resize sync; pointer events must stay with the textarea
  (`WidgetTextarea` stops pointer events in the capture phase and has
  `data-capture-wheel`).

## Grammar the tokenizer must cover

Source of truth is `nodes.py`, not the README (the README omits `<name>` and
`#seed$$`). Comment stripping runs first, per line, cutting at the first `//`
then the first `#`, with `__#` and `<#` protected (`nodes.py` in
`ABWildcardNode.process`).

| Token | Form | Notes |
|---|---|---|
| comment | `// …`, `# …` | not after `__` or `<`; also starts inside `{}`/`[]` and in URLs, because the engine does |
| assignment | `$name = value` at line start | value runs to end of line and is scanned for the other tokens |
| variable | `$name`, `$name.upper` … | methods clean, upper, lower, title, capitalize; unknown methods are ignored by the engine |
| wildcard | `__name__`, `__dir/file__`, `__2$$name__`, `__1-3$$name__`, `__#7$$name__` | glob `*` `?` allowed |
| sequence file | `__~f__`, `__@f__`, `__!f__`, with `k$$` | new in 1.2.0 |
| tag query | `<[ expr ]>` | AND OR NOT XOR (any case), `--` as NOT, parentheses, `$var` operands |
| angle wildcard | `<name>`, `<scope:opts>` | test `<[` before the generic form |
| choice | `{a\|b}`, nested | count `N$$`, `N-M$$`, joiner `N$$ sep $$`, `NN%item`, `W::item` |
| sequence choice | `{~…}`, `{@…}`, `{!…}`, with `k$$` | new in 1.2.0 |
| conditional | `[if cond : a \| b]` | nests; `$var` in the head is an operand, not a substitution |
| function | `[clean: …]`, `[shuffle: …]` | not nested; first `]` closes |

Braces and `[if` need depth counting; a single regex cannot do them.

## Diagnostics

| Severity | Condition |
|---|---|
| error | unclosed `{`, `[if`, `<[`, `[clean:`, `[shuffle:`; unmatched closing `}` or `]>` |
| error | `[if` without `:` |
| warning | `$name` used but never assigned in the text (see Open questions for `globals.yaml`) |
| warning | wildcard or card name or tag not known to the server (uses the list the autocomplete already fetches) |
| warning | `$$` count larger than the option count; empty option list `{}` |
| warning | two identical `{!…}` blocks (they draw the same option, documented in the README) |
| warning | `{@…}` nested inside `{@…}` (not counted evenly, documented) |
| info | `#` or `//` inside a block or URL that starts a comment |

Tune the list after the first pass over the example workflows: a diagnostic that
fires on a valid shipped example is a bug in the rule.

## Steps

1. **Spike (decides the approach).** A throwaway overlay on the existing node
   that only paints one colour for `$name`, in legacy and Nodes 2.0: alignment
   while typing, wrapping, scrolling, node resize, canvas zoom and pan,
   textarea re-creation. Written down: what held, what broke, go or no-go. If it
   cannot be made stable in Nodes 2.0, stop and bring the CodeMirror option back
   to Holger before building more.
2. **Tokenizer** in `js/ab_syntax.js` with the grammar above, plus fixtures
   taken from the README and all `example_workflows/*.json` texts.
3. **Diagnostics** in the same module, rules as in the table.
4. **Painting**: token classes, theme-aware colours (dark and light, both
   renderers), the variable "chip" style, squiggle underline, message on hover or
   strip. Repaint on `input`, coalesced into one animation frame.
5. **Wiring and regressions**: coexistence with the autocomplete popup and the two
   pickers, the refresh button re-validating known names, large prompts.
6. **Docs and release**: README section with a screenshot of the highlighting, the
   grammar-table note in the README (add `<name>` and `#seed$$` while there),
   patch or minor bump per the change, release notes.

## How we'll know it works

1. **Tokenizer test, runnable headless:** `node tests/syntax.test.mjs` (plain
   `assert`, no framework) tokenizes the fixtures and every shipped example text
   with zero diagnostics, and a set of broken inputs yields the exact expected
   diagnostics (for example `{!a|b` gives one error at the `{`).
2. **Both renderers, in a real browser:** with `Comfy.VueNodes.Enabled` on and
   off, typing, undo, paste and autocomplete keep working; the overlay stays
   aligned through scroll, wrap, resize, zoom and textarea re-creation; the
   serialized `text` equals what is typed; no console errors. Evidence is one
   screenshot per renderer, checked with the Read tool.
3. **Existing suite still green:** `pytest -q` passes (the Python side is
   untouched), and typing in a 5,000-character prompt shows no visible lag
   (measured by timing one repaint in the page, not by eye).

## Open questions

- `globals.yaml` variables are loaded server-side only. Either extend
  `/ab_wildcard/wildcards` to return them (small Python change, would belong in
  this plan) or accept unset-variable warnings for them and make that warning
  suppressible. Decide before step 3.
- Exact shape of the tag-query seed form (`<#7$$[…]>`) and `<scope:opts>` was not
  verified in the research; read `nodes.py` before step 2.
- The comparison used frontend 1.58.5 only; recheck the shipping frontend
  version before step 1, since the DOM structure of `WidgetTextarea` matters.
- Prior art not read: `sn0w12/ComfyUI-Syntax-Highlighting` (MIT) highlights
  prompts in multiline boxes. Skim its source for how it handles Nodes 2.0
  before the spike; do not copy code without checking the licence terms.

## Size check

One agent can execute this from the plan and the repo. Step 1 is a gate: if it
fails, the rest of the plan is replaced, not patched.

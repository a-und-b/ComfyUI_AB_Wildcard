# Plan: syntax highlighting and diagnostics for the text widget

Status: step 1 (spike) done, GO; step 2 (tokenizer) done; step 3 (diagnostics) done; step 4 (painting) done; step 5 (wiring) done; step 6 (docs) done, version 1.3.0 set, not released. Written against 1.2.0 (sequences committed).
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
- (Done) `globals.yaml` variable names are served by the endpoint as `globals`.

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
| warning | `$name` used but never assigned in the text (names from the endpoint's `globals` count as assigned) |
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

## Spike result (step 1): GO

Throwaway overlay (mirror `<div>` inserted before the textarea, copy of
computed font/line-height/white-space/wrap, size from `offsetWidth/Height`,
`scrollTop` synced, `ResizeObserver` + `input`/`scroll` listeners), pasted into
the page of the test instance (ComfyUI frontend with Nodes 2.0). Measured by
rect comparison (mirror vs textarea) and equal `scrollHeight` (= identical wrap).

| Case | Nodes 2.0 | Legacy |
|---|---|---|
| Alignment, wrap, scroll sync | held | held |
| Node resize | held | held |
| Canvas zoom 0.6-1.8 and pan | held | held (0 px delta) |
| Real typing, serialized value | held (value equals typed text) | not repeated |
| Textarea re-created (collapse and expand) | **breaks**: new element, mirror gone | element survives |
| Foreign child inside Vue's parent | survives re-render | n/a |

Findings for the build:
- A `MutationObserver` on `document.body` re-attaches after Nodes 2.0 re-creates the textarea (confirmed). One early run did not re-attach within 500 ms; add a `focusin` fallback and make `attach` idempotent.
- In legacy the page holds other, hidden `<textarea>` elements: always select with `resolveTextarea()` (`data-ab-wildcard`), never `querySelector('textarea')`.
- Positioning: absolute mirror inside the textarea's parent works in both (parent is `relative` in Nodes 2.0, `fixed` with CSS transform in legacy, so zoom and pan come for free).
- Textarea needs `background: transparent` and the mirror carries the text colour; `color: transparent` on the textarea with `caret-color` kept is still to try in step 4.
- Not verified: undo (key press through the tool did not change the text, so unproven either way), IME, a classic scrollbar (this browser uses overlay scrollbars, `sb` was 0; the width compensation is written but untested), pointer events and wheel over the overlay, paste, autocomplete popup coexistence (step 5).

## Tokenizer notes (step 2)

`js/ab_syntax.js` + `tests/syntax.test.mjs` (`node tests/syntax.test.mjs`): 12
shipped example texts give zero diagnostics. Structural errors (unclosed and
unmatched `{ [if [clean: <[`, `[if` without `:`) are already in; the warnings of
step 3 are not. For step 3:
- `<name>` is only marked when it has no inner whitespace, but `<lora:x:0.8>` still
  becomes an `angle` token, so "unknown wildcard" must skip it (the engine keeps it literal).
- `$5`-style prices match the engine's variable regex; warn on unset only for names starting with a letter or `_`.
- Tag-query seed form `<#7$$[…]>` is tokenized as a tag query; `#7$$` is plain text inside.

## Diagnostics notes (step 3)

`tokenize(text, known)` takes the endpoint data `{ wildcards, tags, globals }`;
without it the name rules stay silent. All rules of the table are in, tested in
`tests/syntax.test.mjs`, and all 12 shipped example texts give zero diagnostics,
also against the live endpoint lists. Deliberate limits:
- "`$name` never assigned" is **silent in any text that uses a wildcard, card or tag query**: CSV columns and cards can define variables the frontend cannot see. Closing that gap would need the endpoint to return CSV column names; not planned.
- Unknown-name warnings cover `__name__` and the tags in `<[ … ]>`; `<name>` is not checked (indistinguishable from `<lora:…>`). Wildcard names match case-insensitively, by full key or by `/name` suffix.
- Count rule compares the lower bound of `N` / `N-M` with the option count.

## Painting notes (step 4)

`js/ab_highlight.js` (`installHighlighter`, wired in `setup()` of `js/umi_wildcards.js`;
`fetchWildcards` now keeps `globals` and repaints). Checked in the Browser pane:
both renderers show colours, variable chips, wavy underlines and the status strip;
light palette is readable; re-creation after collapse re-attaches with one mirror;
selection highlight lines up with the mirror text (legacy). One repaint of a
5,280-character prompt: tokenize 0.3 ms, render 0.6 ms, DOM and layout 8.8 ms.
- Messages: shown in a strip at the bottom of the field (counts + the diagnostic at the caret, else the worst one). Hover is not possible, the mirror ignores the pointer.
- Repaint is coalesced with `requestAnimationFrame` plus a 60 ms timeout, because rAF is paused in a hidden page.
- Theme: light/dark is picked at attach and on `repaintAll`, not on a live palette switch.
- Server needs a restart for the `globals` field; without it the highlighter works, `globals` is just empty.

## Wiring notes (step 5)

Checked in Nodes 2.0 by DOM state (mirror text equals textarea value, token classes present):
programmatic `el.value = x` (now repaints through an instance-level `value` setter hook, since Vue's v-model and
workflow load fire no `input`), widget-driven set, picker insert, paste (`insertText`), undo (`execCommand('undo')`),
refresh (`fetchWildcards` repaints), autocomplete (popup shows with 2 suggestions, selecting inserts `__Color__`,
chip class appears, caret at end). Not covered: key-driven undo through the tool (unsupported), IME, a real
mouse wheel over the field, legacy renderer for steps 5's cases. Screenshots from the hidden pane were stale after
the first ones, so the popup was checked by state and geometry, not by picture.

## Open questions

- ~~`globals.yaml` variables~~ Decided: `/ab_wildcard/wildcards` and `/refresh` now return `globals` (variable names only, no values). The unset-variable warning treats them as defined.
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

# Plan: find and replace in the text field

Status: steps 1-2 done; 3-4 open. Written against 1.4.0 (branch
`feat/autocomplete-variables`, not yet released).
Format note: this repo has no `.ab/`; the plan lives here like the other
`docs/plan-*.md`, so any session can pick it up.

## Goal

Inside the AB Wildcard node's `text` widget, in both renderers, Cmd/Ctrl+F
opens a small find bar on the field. Observable outcome: type `color` and
every match in the field is marked, the current one stronger, with "2 / 5";
Enter / Shift+Enter jump to the next / previous match and scroll it into view;
Replace and Replace all swap the text, the widget value is synced, and one
Cmd/Ctrl+Z undoes the last replace. Esc closes the bar and returns focus to the
field. Outside the field, Cmd/Ctrl+F behaves exactly as before.

## Facts this plan rests on

- The highlighter already repaints the text in a mirror `<div>` behind the
  transparent textarea (`js/ab_highlight.js`, `render(text, {tokens,
  diagnostics})`). Match marks are one more kind of range there; no second
  overlay.
- The status chip (`.ab-hl-status`) shows how a small element is placed inside
  the textarea's parent in both renderers; the find bar sits the same way.
- `textarea.value = …` throws away native undo. `document.execCommand(
  "insertText")` on a selection keeps it (Chromium, which both the browser and
  the desktop app use) and fires `input`, which syncs the widget.
- The frontend ships no default Cmd/Ctrl+F binding (grep of the bundled
  assets); the key is free once the field has focus.

## Decisions (agreed with Holger)

1. **Two plans.** This one is the field. Searching wildcard and card
   *contents* needs a server endpoint and is the next plan
   (`plan-wildcard-search.md`), reusing this bar with a mode switch.
2. **Find and replace**, with Replace and Replace all.
3. **Opened by Cmd/Ctrl+F while the field has focus**, nothing permanent on
   the node.

Defaults (no decision needed, change if wrong):

4. Case-insensitive plain text; no regex, no whole-word toggle.
5. Replace all is one undo step: select all, `insertText` the new text.
6. Match finding is a pure function in `ab_syntax.js`
   (`findMatches(text, query)` → `[{start, end}]`, plus the replace-all
   result), tested in `tests/syntax.test.mjs` like the tokenizer.

## Not in scope

- Searching wildcard files and cards (next plan).
- Regex, case or whole-word toggles, search history.
- Searching across several nodes or the whole workflow.
- Any change to `nodes.py`.

## Steps

1. **Pure functions + tests.** `findMatches` and `replaceAll` in
   `js/ab_syntax.js`; cases in `tests/syntax.test.mjs`.
2. **Marks in the mirror.** `render()` takes optional match ranges and the
   current index; CSS classes for a match and the current match, readable in
   light and dark.
3. **Find bar.** Key handling (Cmd/Ctrl+F only on our textarea, Enter,
   Shift+Enter, Esc), the bar with find field, replace field, count, next,
   previous, Replace, Replace all; scroll the current match into view;
   replace through `insertText`. Both renderers.
4. **Docs.** README section and the step list near the end; version 1.5.0 in
   `pyproject.toml`, not released.

## How we'll know it works

1. `node tests/syntax.test.mjs` passes with cases that fail today because the
   functions do not exist: `findMatches("Red red RED", "red")` gives three
   ranges; an empty query gives none; `replaceAll("$color $Color", "$color",
   "$tint")` gives `"$tint $tint"`; a query containing `$` or `{` is matched
   literally.
2. In a running ComfyUI, Nodes 2.0 and Legacy, with real key presses: Cmd/Ctrl+F
   in the field opens the bar, typing marks all matches and shows the count,
   Enter moves the current mark and scrolls, Replace all changes the text and
   the serialized widget value, one Cmd/Ctrl+Z restores it, Esc closes and the
   caret is back in the field.
3. Same session: Cmd/Ctrl+F with the canvas focused (not the field) does not
   open the bar, and autocomplete, highlighting and the pickers still work
   with the bar open and after it closed.

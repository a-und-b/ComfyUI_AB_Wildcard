# Plan: search wildcard and card contents from the node

Status: step 1 built (endpoint + tests), steps 2 and 3 open. Written against 1.5.0 (find and replace in
the field, `plan-text-search.md`).
Format note: this repo has no `.ab/`; the plan lives here like the other
`docs/plan-*.md`, so any session can pick it up.

## Goal

Inside the AB Wildcard node's `text` widget, in both renderers, the Cmd/Ctrl+F
find bar has a mode switch "Text | Wildcards". Observable outcome: in
Wildcards mode, type `emerald` and a list appears under the bar with one row
per insertable token, here `__examples/colors__` and `__Shades Of Green__`,
each with the first matching line as a preview and the number of further hits
("emerald" +0). Arrow keys or the mouse pick a row; Enter or a click inserts
the token at the caret, closes the bar and leaves the caret after the token;
one Cmd/Ctrl+Z removes it again. Text mode behaves exactly as in 1.5.0.

## Facts this plan rests on

- `TagLoader.build_index()` puts every insertable name into `files_index`:
  txt and csv keys (`examples/colors`), card titles (`Iron Knight`) and leaf
  paths of hierarchical yaml (`file/a/b`). `globals.yaml` is skipped. These are
  exactly the names autocomplete offers, so they are the tokens to return.
- `TagLoader.load_tags(key)` already reads any of these keys (txt lines, csv
  rows as dicts, yaml leaves) and caches them in `GLOBAL_CACHE`; the search
  needs no file reading of its own. For a card title it returns `[title]`, so
  card contents come from `yaml_entries[title.lower()]` (`prompts`,
  `description`, `tags`) instead.
- `GLOBAL_CACHE` and the index are cleared by `POST /ab_wildcard/refresh`
  (the node's refresh button), so edited files show up in search after a
  refresh — the same rule autocomplete already follows.
- The find bar (`js/ab_highlight.js`, `openFind` / `closeFind`) is one `.ab-find`
  element per textarea with a find row and a replace row; `insert(t, start,
  end, text)` goes through `execCommand("insertText")`, which keeps native undo
  and fires `input`, so the widget value syncs.
- While focus is in the bar the textarea keeps its `selectionStart/End`, so the
  caret position to insert at is still known when the row is picked.
- The shipped `wildcards/examples` contain `emerald` in `colors.txt` and in the
  card `Shades Of Green` (`color-shades.yaml`), and `knight` in the title,
  a prompt and a tag of `Iron Knight` — usable as test data.

## Decisions (agreed with Holger)

1. **Separate plan from 1.5.0**, reusing its bar with a mode switch and a new
   endpoint next to `/ab_wildcard/wildcards` (agreed when 1.5.0 was planned).
2. **One result per token**, not per line: token, first matching line as the
   preview, count of further hits; at most 50 results.
3. **Cards are searched in title, prompts, description and tags.** Files in
   their name and every line (csv: every cell).
4. **Picking a result inserts `__token__` at the caret, closes the bar, caret
   after the token**, one undo step.

Defaults (no decision needed, change if wrong):

5. Case-insensitive plain substring, like Text mode. Queries shorter than 2
   characters return nothing.
6. Ordering: tokens whose *name* matches first, then content matches, each
   group alphabetical. No scoring.
7. `GET /ab_wildcard/search?q=…` → `{"results": [{"token", "kind": "file" |
   "card", "preview", "more"}]}`. The matching is a plain function
   `search_wildcards(loader, query, limit=50)` in `nodes.py`, tested in
   `tests/test_processing.py`; the route only builds the loader and calls it.
8. The frontend asks on input with a short debounce (~150 ms) and drops
   answers to stale queries. The bar opens in the mode it was last closed in.
9. In Wildcards mode the replace row is hidden; the result list takes its
   place. Esc closes the bar as before.

## Not in scope

- Inserting `<[tag]>` for tag matches (autocomplete covers tags), or a card's
  individual prompt instead of the card token.
- Regex, fuzzy matching, ranking, search history.
- Editing wildcard files from the node, or opening them.
- A persistent search index or file watching; refresh stays the only reload.
- Searching other nodes or the workflow.

## Steps

1. **Backend + tests.** `search_wildcards()` and the `/ab_wildcard/search`
   route in `nodes.py`; cases in `tests/test_processing.py`.
2. **Mode switch and result list.** In `js/ab_highlight.js`: a "Text |
   Wildcards" toggle in the bar, the result list (token, preview, `+n`),
   debounced fetch, arrow/Enter/click selection, insert through `insert()`
   and close. The fetch is passed in from `umi_wildcards.js` like
   `getKnown`, so the highlighter does not learn URLs. Both renderers; list
   readable in light and dark.
3. **Docs.** README section on the Wildcards mode next to find and replace;
   version 1.6.0 in `pyproject.toml`, not released.

## How we'll know it works

1. `pytest tests/test_processing.py` passes with cases that fail today because
   the function does not exist: `emerald` returns `examples/colors` (file) and
   `Shades Of Green` (card); `knight` returns `Iron Knight` once, listed
   among the name matches; a one-character query returns nothing; a query
   matching nothing returns an empty list.
2. In a running ComfyUI, Nodes 2.0 and Legacy, with real key presses and
   clicks: Cmd/Ctrl+F, switch to Wildcards, type `emerald`, the two rows
   appear; Arrow Down + Enter inserts `__Shades Of Green__` at the caret, the
   bar closes, the serialized widget value contains it, one Cmd/Ctrl+Z removes
   it.
3. Same session: switching back to Text mode still finds and replaces as in
   1.5.0, and after adding a line to a wildcard file plus the node's refresh
   button, a search for that line finds it.

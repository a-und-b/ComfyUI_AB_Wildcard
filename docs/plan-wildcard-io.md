# Plan: wildcard file I/O and text operator nodes (AB Wildcard 1.7.0)

Status: UX rework built on branch `feat/wildcard-io` (version 1.7.0), not merged
or released. Checked: pytest (67) and syntax tests green; live in the test ComfyUI
(0.45.2, frontend 1.58, Nodes 2.0): every operation runs with the expected output,
a full graph survives save → reload with every widget value intact, both example
workflows run. Not checked: the Legacy (non-Vue) renderer; Holger's hand test.

Found while building (frontend 1.58):
- Socket-only inputs (`forceInput`) next to a seed widget, or ahead of a
  DynamicCombo, make saved widget values land in the wrong widget or reset on
  load. Hence: Text Lines `text`, Save `text` and the Prompt Mix pieces are text
  widgets (still linkable), not sockets.
- Fields inside a DynamicCombo show their id, not `display_name`, so those ids
  are the readable labels (`keep lines that`, `chance 1 %`).
- `DynamicCombo.Input` takes no `advanced` argument; it goes through `extra_dict`.
- V3 classes register fine through the existing `NODE_CLASS_MAPPINGS`.
- Save mode labels became plain words: add new lines · add all lines · replace file.
Format note: this repo has no `.ab/`; the plan lives here like the other `docs/plan-*.md`.

## UX rework (agreed with Holger 2026-11-21, built) — supersedes the node specs below

Why: the built nodes work but are unpleasant to use: AB Text Lines shows 11
fields for a one-field job, regex and `\n` escapes leak developer language, the
example needs a regex trick, Load shows a `file` that its `name` input silently
overrides, and Prompt Mix's `weight` means two different things. Audience is
both Holger (power user) and non-developer creatives: plain defaults, expert
fields folded away.

Agreed decisions:
1. **V3 node API** (`comfy_api.latest.io.ComfyNode`) for the four new nodes;
   AB Wildcard itself stays V1. The pack then needs a current ComfyUI
   (tested: 0.45.2, frontend 1.58 — also what dd-machine runs).
2. **AB Text Lines = one node, `operation` dropdown (DynamicCombo)**; only the
   chosen operation's fields show. Several steps = several nodes in a row.
   - sort: `direction` (A → Z, Z → A, reverse order)
   - filter: `keep lines that` (contain, don't contain, start with, end with,
     are exactly, match regex) + `text`
   - replace: `find`, `replace with`; `use regex` under advanced
   - remove duplicates: no fields (`ignore case` under advanced)
   - shuffle: `seed`
   - split and join: `split at` / `join with`
   Lines are always trimmed and empty ones dropped.
3. **No regex by default**: plain matching, any case; regex is the last
   `keep lines that` option and an advanced toggle on replace.
4. **Separators as a dropdown**: new line, comma, space, custom (a text field
   appears only for custom). No `\n` escapes in the UI. Outside "split and join"
   they sit under advanced, default new line.
5. **Save shows the whole file**: `text` is a multiline widget (type a list or
   link one), outputs are the file's full content after saving: `text`, `lines`,
   `count`. Load drops its `name` input and keeps only the `file` dropdown.
6. **Prompt Mix**: `mode` as DynamicCombo — "some pieces (each by chance)" shows
   `chance 1–4` in % (0–100, default 100); "one piece (weighted)" shows
   `weight 1–4`. Four fixed text sockets. Separator dropdown as in 4.
7. **Two small examples instead of one big one**: "Build a list" (Save with a
   typed list → Preview; Load → Text Lines sort → Preview) and "Mix pieces"
   (three Text nodes → Prompt Mix → AB Wildcard → Preview). No regex trick.

Technical facts checked in the test install (0.45.2 / frontend 1.58):
- `DynamicCombo`, `Autogrow` and the `advanced` input flag exist in
  `comfy_api/latest/_io.py`; the frontend hides `advanced` inputs behind a
  toggle and handles `COMFY_DYNAMICCOMBO_V3`.
- `Autogrow` takes exactly one template input and turns widgets into sockets,
  so it cannot carry a weight per slot; `DynamicSlot` (slot + extra fields)
  exists in Python but the frontend 1.58 bundle does not know it. Hence four
  fixed slots in Prompt Mix.
- ComfyUI's loader takes `NODE_CLASS_MAPPINGS` first; `io.ComfyNode` exposes a
  V1-compatible `INPUT_TYPES`, so V1 and V3 classes should share the mapping.
  **Verify first when building** (if not: register V3 nodes via `comfy_entrypoint`
  in a second module).
- Tests run without ComfyUI: pure logic moves to `ops.py` (no ComfyUI imports,
  tested by pytest), `tools.py` keeps thin V3 wrappers. The roundtrip check
  (serialize → configure keeps every widget value) runs in the test instance.

Not doing: own JS widgets (the built-in dynamic inputs cover it; JS would break
on frontend updates), growing slot count in Prompt Mix (needs DynamicSlot).

## Context

Today wildcard files can only be made by hand in an editor; the AB Wildcard node
only reads them. Holger wants to build and read wildcard files inside a graph
(e.g. an LLM or a list node produces lines → save as `__name__` → reuse), plus a
small set of text nodes for shaping lines and for steering which prompt pieces
flow into a prompt, and how often. Outcome: four new nodes in this pack,
category `AB Utils`, next to `AB Wildcard`.

Agreed with Holger (2026-11-21):
- Save target: `ComfyUI/wildcards` (already read by `get_all_wildcard_paths`,
  survives pack updates; created on first save).
- Save modes: `overwrite` / `append` / `append unique`.
- Load outputs: text + list + count. Load returns raw content, never a random
  pick — random picking is what `__name__` in AB Wildcard already does.
- Text operators: as few nodes as possible covering trim/dedupe/sort/shuffle,
  join, replace/filter, split → list; plus a node that controls the inflow of
  prompt pieces by probability/weight.

## Facts this plan rests on

- `get_all_wildcard_paths()` (`nodes.py:187`) adds `folder_paths.base_path/wildcards`
  only if it exists → after the first save it is picked up automatically.
- `TagLoader.refresh_maps()` (`nodes.py:330`) keys files as `rel/path` without
  extension, lower-cased, `/` separators → same key scheme for the Load dropdown.
- `read_file_lines()` (`nodes.py:59`) strips empty lines and `#` comments → reuse
  for Load and for the `append unique` comparison.
- Caches: `GLOBAL_CACHE` and `GLOBAL_INDEX` (`nodes.py:14-16`); the refresh route
  (`nodes.py:1435`) clears them the same way → a saved file must clear them too,
  so AB Wildcard sees it in the same session without pressing Refresh.
- Tests stub `server`/`folder_paths` in `tests/conftest.py`; `folder_paths.base_path`
  can be monkeypatched to `tmp_path`.

## The four nodes (new file `tools.py`, registered in `__init__.py`)

### 1. AB Wildcard Save (`OUTPUT_NODE`)
- Inputs: `text` (STRING, forceInput), `name` (STRING, e.g. `my/colors`),
  `mode` (combo: overwrite · append · append unique).
- Splits `text` into lines (trim, drop empty), writes `ComfyUI/wildcards/<name>.txt`
  (creates subfolders). `append unique` skips lines already in the file and
  duplicates within the input.
- **Path guard** (trust boundary): reject absolute names, `..`, and anything whose
  resolved path leaves the wildcards dir; strip a trailing `.txt`.
- Clears `GLOBAL_CACHE` / `GLOBAL_INDEX` after writing.
- Outputs: `text` (pass-through) and `name` (so Save → Load can be chained).
- Note: ComfyUI caches unchanged nodes, so re-queuing identical input does not
  append twice. That is the desired behaviour; documented, no `IS_CHANGED`.

### 2. AB Wildcard Load
- Inputs: `file` (combo of `.txt` keys from all wildcard paths, via `TagLoader`
  maps), optional `name` (STRING input) that overrides the combo when connected.
- Reads with `read_file_lines`. Unknown name → clear error naming the file.
- Outputs: `text` (lines joined by `\n`), `lines` (`OUTPUT_IS_LIST`), `count` (INT).
- ponytail: `.txt` only; csv/yaml when someone needs them.

### 3. AB Text Lines (one node for all line operations)
Fixed pipeline, each step a no-op at its default:
split by `split_by` → find/replace → trim + drop empty (always) → keep lines
matching `keep` → drop lines matching `drop` → `dedupe` → `order` → join with `join_with`.
- Widgets: `text` (forceInput), `split_by` (default `\n`; `\n`/`\t` escapes
  understood), `find`, `replace`, `keep`, `drop`, `regex` (BOOLEAN, applies to
  find/keep/drop, else case-insensitive plain substring), `dedupe` (BOOLEAN, default on),
  `order` (combo: as is · sort · sort desc · reverse · shuffle), `seed` (for shuffle),
  `join_with` (default `\n`).
- Invalid regex → readable error naming the field.
- Outputs: `text`, `lines` (list), `count`.
- Covers: trim/dedupe/sort/shuffle, join, replace/filter, split → list.

### 4. AB Prompt Mix (inflow of prompt pieces)
- Inputs: `mode` (combo: `each by chance` · `pick one weighted`), `seed`,
  `separator` (default `, `), and four optional slots `text_1..4` (STRING, forceInput)
  each with `weight_1..4` (FLOAT 0–1, default 1).
- `each by chance`: every connected, non-empty slot is included with probability
  = its weight; included pieces joined in slot order.
- `pick one weighted`: exactly one non-empty slot, chosen by relative weight.
- Local `random.Random(seed)` — never touches the global RNG AB Wildcard seeds.
- Output: `text`.
- ponytail: four fixed slots; dynamic growing inputs (needs JS) when four is too few.

## Files

- `tools.py` — new: the four nodes + helpers (`_wildcards_dir`, `_safe_path`, `_lines`).
  Imports `read_file_lines`, `TagLoader`, `get_all_wildcard_paths`, caches from `nodes.py`.
- `__init__.py` — add the four classes to both mappings.
- `tests/test_tools.py` — new, pytest.
- `README.md` — section "Wildcard files in the graph" + "Text tools"; version note.
- `pyproject.toml` — version 1.7.0.
- `example_workflows/ab-wildcard-tools.json` — Save → Load → Text Lines → Prompt Mix
  → AB Wildcard → Preview as Text, no models; screenshot if a local ComfyUI runs.
- `docs/plan-wildcard-io.md` — this plan, copied in at build start (repo convention:
  no `.ab/`, plans live in `docs/plan-*.md`).

## Build order (one commit each, conventional messages as in history)

1. Copy plan to `docs/`, commit `docs: plan for wildcard file nodes and text tools`.
2. Save + Load + path guard + tests → `feat(io): save and load wildcard files from the graph`.
3. Text Lines + tests → `feat(tools): AB Text Lines`.
4. Prompt Mix + tests → `feat(tools): AB Prompt Mix`.
5. README, example workflow, version 1.7.0 → `docs: …`.

## Verification

- `pytest` — new tests:
  - Save: overwrite / append / append unique against `tmp_path`; `../x`, `/etc/x`,
    `a/../../x` rejected; caches cleared after save; Load reads back what Save wrote.
  - Text Lines: each step alone, regex vs plain, shuffle stable per seed, bad regex error.
  - Prompt Mix: weight 0 never included, weight 1 always; weighted pick deterministic
    per seed and never picks an empty slot.
- `pytest` existing suite and `node tests/syntax.test.mjs` stay green.
- Live: start local ComfyUI (comfy-mcp `server_info` / `launch_comfyui`), load the
  example workflow, queue: file appears in `ComfyUI/wildcards/`, `__name__` in the
  AB Wildcard node resolves it without Refresh, Preview shows expected text;
  Load dropdown lists it after a browser reload.

## Not in this round

csv/yaml save or load, YAML card writing, dynamic slot count, a delete-file node.

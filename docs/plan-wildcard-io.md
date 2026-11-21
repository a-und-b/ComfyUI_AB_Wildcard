# Plan: wildcard file I/O and text operator nodes (AB Wildcard 1.7.0)

Status: built on branch `feat/wildcard-io`, version 1.7.0, not merged or released.
Checked: pytest (65) and syntax tests green; live in the test ComfyUI (frontend
1.58, Nodes 2.0) — example workflow saves, reloads with correct widget values,
runs: Load → "amber, coral, teal", `__my/colors__` resolved without Refresh.
Not checked: the Legacy (non-Vue) renderer; Holger's hand test.

Deviation found while building: AB Text Lines takes `text` as a multiline widget,
not `forceInput`. A forceInput socket next to a `seed` widget made frontend 1.58
restore every saved widget value one slot too early (reproduced with test node
defs; Save, Load and Prompt Mix are not affected). The widget can still be linked.
Format note: this repo has no `.ab/`; the plan lives here like the other `docs/plan-*.md`.

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

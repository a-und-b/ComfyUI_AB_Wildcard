# Plan: big prompt → wildcard files (AB Wildcard Extract)

Status: built 2026-11-23 on `feat/wildcard-extract` (`ops.extract`,
`ops.save_extracted`, `tools.ABWildcardExtract`, `tests/test_extract.py`).
Example workflow, preview and README section added on `release/1.11.0`
(not pushed), together with AB Image from Folder (`plan-image-from-folder.md`).

## Goal

Turn a big prompt — or a long text of lists, such as LLM output — into
wildcard files in one step, and get the prompt back with `__…__` references in
place of the inline lists. The lists become reusable, editable files, and the
prompt becomes short and readable.

## Interface

ID `ABWildcardExtract`, display `AB Wildcard Extract`, same category as
AB Wildcard Save/Load (V3 `io.ComfyNode` in `tools.py`).

- Inputs: `text` (multiline text widget, linkable), `folder` (default
  `extracted`), `mode` = `ops.SAVE_MODES`, default **`replace file`**.
- Outputs: `text` (rewritten prompt), `files` (one `folder/name` per line),
  `count` (files written).
- Output node like Save; ComfyUI skips it when the inputs didn't change.

Example:

```text
in:   $color = {red|blue|green}
      a $color car in {Paris|Tokyo|Berlin}, {~morning|noon|night}

      Moods:
      - calm
      - stormy

out:  $color = __extracted/color__
      a $color car in __extracted/choice_1__, __~extracted/choice_2__

      __extracted/moods__
```

## Decisions (Holger, 2026-11-23)

- **One node, both input forms, one pass.** List sections first (line based),
  then inline blocks in the remaining text.
- **List sections**: a heading line — `Name:`, `# Name` / `## Name`, or
  `**Name**` — followed by items (`- `, `* `, `1. `, or plain lines) until an
  empty line or the next heading. The list markers are stripped. The section
  is replaced by `__folder/<slug>__`. A heading with no items is left alone.
- **Inline blocks**: only top-level `{…}` blocks. A nested block stays as text
  inside its line (file lines are resolved recursively — verify, see Checks).
  Options split at top-level `|` only.
- **Names**:
  - `$name = {…}` → the variable name; the line becomes `$name = __folder/name__`.
  - Unnamed blocks → `choice_1`, `choice_2` … by position in the prompt, so the
    same prompt gives the same names. README recommends `$name = {…}`.
  - Headings → slug: lower case, spaces to `_`, only letters, digits, `_`,
    `-`; empty slug → `list_N`. A name already used in this run gets `_2`,
    `_3`. Names must not start with `~ @ !`; paths go through `ops.safe_path`.
- **Prefixes carry over**: `{~…}` → `__~…__`, `{!…}` → `__!…__`,
  `{@…}` → `__@…__`, `{2$$…}` / `{1-3$$…}` → `__2$$…__` / `__1-3$$…__`.
- **Left inline, with one console warning listing them**: weighted `{5::a|b}`,
  chance `{50%a|b}`, custom separators `{2$$ and $$…}`, blocks with an option
  containing `#` (cut as a comment in files), blocks with a single option.
- **Mode `replace file` by default**: re-running an edited prompt keeps the
  files in sync with it. Safe because the node only writes inside `folder`.
- Files are usable right away (`ops.clear_wildcard_caches()`), like Save.

## Checks

- pytest: both forms; `$name =` naming; positional `choice_N` stable across
  runs; every prefix (`~ ! @ $$`, ranges); nested block stays one file line;
  each skipped kind stays inline and is reported; slug rules and `_2`
  collisions; `..` or absolute `folder` refused; the three modes.
- Resolve check: a file line holding `{a|b}` resolves through `__name__` the
  same way as inline (decides whether nested blocks are safe to move).
- Roundtrip: the rewritten prompt through AB Wildcard gives the same option
  set as the original (same seed, `~` sequences identical).
- Widget roundtrip (NODE_MAINTENANCE §3); example workflow + preview.

## Found while building

- Resolve check passed: a file line holding `{a|b}` resolves through
  `__name__` exactly like inline, so nested blocks move into files.
- Two more kinds stay inline (a file can't express them): a block with an
  empty option (`{a|}` means "a or nothing"; files drop empty lines), and a
  `$$` block whose count is not a number or range (`{x$$a|b}`).
- `{N%…}` counts as chance only when the text before `%` is a number, as in
  AB Wildcard (`{100% cotton|silk}` is a chance block there too).
- Roundtrip: `~` and `@` give the same prompt per seed; `!` gives the same
  option set but a different order, since its shuffle is salted with the
  block's text inline and with the file name in a file.

## Left out

- Dry run without writing (Preview as Text shows the prompt already).
- Weights written as repeated lines (`add new lines` would dedupe them).
- YAML cards as output; choosing names per block in the UI.
- A frontend button on AB Wildcard that extracts in place (needs JS; the node
  covers it for now).

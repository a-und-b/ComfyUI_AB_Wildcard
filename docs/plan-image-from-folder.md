# Plan: images from a folder (AB Image from Folder)

Status: built 2026-11-23 on branch `feat/image-from-folder` (`folder.py`,
`tests/test_folder.py`); pytest and the widget roundtrip (both renderers,
seed linked or not) pass on the local test engine. Not released; the live
checks on dd-machine, README and example workflow are still open. Origin: analysis of
[ComfyUI-DirGir](https://github.com/AshMartian/ComfyUI-DirGir) — its folder
loop is the useful part, its state handling is not (see "Why not DirGir").

## Goal

One node that loads one image per run from a folder, for three jobs:

1. **Batch a folder** — every image of a shoot through the same workflow.
2. **Fixed test set** — the same images, in the same order, to compare models,
   LoRAs or samplers; combined with `{@…}` in AB Wildcard, every pairing
   exactly once.
3. **Random style reference** — a different reference per run, without repeats
   until every image was used.

User-friendly over feature breadth: one node, one `pick` field, no state.

## Why not DirGir

- The index lives in server RAM and is incremented by the frontend
  (`afterQueued`); a restart starts over at file 0, and a run queued through
  the API or comfy-mcp sends unchanged inputs, so ComfyUI most likely serves
  the cached result again (not tested).
- `random` reshuffles on every run, so images repeat and others never come.
- The directory picker opens a tkinter dialog on the *server*; ComfyUI runs on
  dd-machine, so the dialog would open there.
- It installs tkinter with `sudo apt` / `brew` at import time.

## Decisions

- **Pack: AB Wildcard** (Holger, 2026-11-23). The index comes from
  `SequenceContext` in `nodes.py`, the same seed logic as `{~ ! @}`; the node
  shares the seed concept, and can share the seed link, with the prompt.
- **One node, not three.** The jobs differ only in how the file is chosen.
- **Index from the seed alone**, nothing remembered between runs. Same seed,
  same image; ComfyUI caching and API runs behave.
  - `in order`: `(seed // every) % count` (= `SequenceContext.step`)
  - `shuffled, no repeats`: `deck_at(seed // every, count, salt=folder)`
  - `random`: own `random.Random(f"ab_folder|{seed // every}")`, never the
    global stream (`every` holds the image in all three picks)
- **`every` (advanced, default 1)**: next image only after N runs. For job 2 the
  image turns slowest: 5 images × `{@a|b|c}` in the prompt → `every: 3`,
  seed linked to AB Wildcard, 15 runs.
- **Folder**: a plain text widget (linkable). Relative paths resolve against
  `folder_paths.get_input_directory()` and must stay inside it (`..` refused);
  absolute paths are allowed (LAN setup, same as VHS path loaders).
- **Files**: `match` (advanced) takes `;`-separated globs, case-insensitive,
  default `*.png;*.jpg;*.jpeg;*.webp`. Not recursive. Hidden files skipped.
- **Sort** (advanced): `name` with natural order (`img2` before `img10`), or
  `date modified`.
- **Loading** as core Load Image: EXIF transpose, RGB, mask from alpha
  (`1 - alpha`, zeros without alpha). Multi-frame files: first frame only.
- **Outputs**: `image`, `mask`, `file_name` (stem, for save names and
  captions), `file_path`, `index` (0-based, matches the seed), `count`.
- **Cache**: `fingerprint_inputs` hashes the matched file names, mtimes and
  sizes, so an added or changed file triggers a re-run.
- **Errors**: missing folder or no matching files stop the run with the
  resolved path and the `match` pattern in the message.
- **Names**: ID `ABImageFromFolder`, display `AB Image from Folder`, category
  `AB/Loaders` (scheme in `../../NODE_MAINTENANCE.md` §6; the pack's other
  nodes move from `AB Utils` in the cleanup release).
- **Widget layout**: no socket-only inputs next to `seed` (frontend 1.58 load
  bug, see `plan-wildcard-io.md`); serialize → `loadGraphData` roundtrip
  before release.

## Checks

- pytest (`tests/`): index math for all three picks and `every`; deck gives
  every index once per round; `random` reproducible per seed; `..` refused;
  natural sort; `match` filter; empty folder error; fingerprint changes when a
  file is added or touched.
- Live on dd-machine: folder of 5 images, `{@a|b|c}` prompt, `every: 3`, seed
  linked, 15 runs → 15 distinct pairings. Restart ComfyUI mid-batch → the next
  run continues at the right image. One run queued via comfy-mcp
  `run_workflow` with seed +1 → next image.
- Widget roundtrip (NODE_MAINTENANCE §3); example workflow + preview on the
  grid.

## Open

- ~~Can the V3 `seed` input default to `increment`?~~ Yes:
  `control_after_generate=io.ControlAfterGenerate.increment`. Frontend 1.58
  takes a string value as the default mode (`randomize` only for `True`);
  checked live: a new node shows `increment`, three queued runs gave seeds
  0 → 1 → 2 and img1 → img2 → img3.
- Getting images onto dd-machine (`input/<folder>`) is out of scope here;
  README names comfy-mcp `upload_file` and ab-remote.

## Left out

- Directory picker dialog (server-side, useless on a remote ComfyUI).
- A "queue whole folder" button that sets batch count to `count` (needs JS).
  Candidate for later.
- Subfolders, video files (VHS Load Video (Path) takes `file_path`), text or
  caption files next to the image, Eagle as source instead of a folder.

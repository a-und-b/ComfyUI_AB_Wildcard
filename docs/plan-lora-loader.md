# Plan: LoRAs from the prompt (AB LoRA Loader)

Status: merged (PR a-und-b/ComfyUI_AB_Wildcard#10), not released yet.
Checked live on dd-machine (2026-11-22, ComfyUI 0.46.2, juggernautXL +
`SDXL/Standard/princess_xl_v2`, same seed): with vs without the LoRA differ
clearly (mean pixel diff 38/255); `<LORA:sdxl/standard/Princess_XL_v2.safetensors:1>`
gives the pixel-identical image to `<lora:princess_xl_v2:1>`; a missing name
stops with a suggestion, a name in two folders lists both. Found there and
fixed: `if_missing` had no default, so API-format graphs without it failed
validation; "Did you mean" showed the lower-cased path instead of the name
autocomplete would write.

## Goal

`<lora:name:strength>` in the prompt, as in A1111/Forge, with the pack's own
strengths on top: autocomplete for LoRA files, a warning for unknown names,
and everything the wildcard syntax already does (random choice, `[if …]`,
variables, card prefixes) working on the tags. User-friendly and robust over
feature breadth.

## Decisions

- **AB Wildcard stays pure text.** It already passes `<lora:…>` through
  untouched (tested since 1.x). Loading happens in a separate node, **AB LoRA
  Loader**: `model`, `clip` (optional), `text` → `model`, `clip`, `text`
  without tags, `loras` (what was applied). It also works after any other text
  source.
- **Keyword triggers need no new feature**: `[if cyberpunk : <lora:neon:0.6>]`
  does what "String Contains → LoRA switch" wiring does in other packs. A card
  brings its LoRA via `Prefix:`.
- **Syntax**: `<lora:name>`, `<lora:name:m>`, `<lora:name:m:c>`; numbers are
  taken from the right, so a name may contain `:`. `<lyco:…>` is accepted.
  A tag without a name stops the run with the tag in the message.
- **Names**: case-insensitive, extension optional, `/` or `\`. Full relative
  path wins, then path without extension, then the bare file name. A bare
  name found in several folders is an error that lists them (no silent pick).
  Not found: error with up to three close names (difflib). Autocomplete
  inserts the shortest unique form. Rules live in `loras.py` and are mirrored
  in `js/ab_syntax.js` (`loraMatches`, `loraInsertNames`); both have tests.
- **Missing LoRA**: default *stop with error* (a typo otherwise costs a whole
  batch of wrong images); *skip it* under advanced, with a console warning.
- **Same LoRA twice**: strengths add up (same result as loading twice, one
  patch). Strength 0 is skipped. No clip input: clip strength is ignored.
- **Text cleanup** only on lines that held a tag; a line of only tags goes.
  Other lines are not touched, so the node is safe on any text.
- **`#` inside a tag** is part of the name in AB Wildcard's comment cut and in
  the highlighter (before, `<lora:v#2>` was cut to `<lora:v`).
- **Loading** goes through ComfyUI's own `load_torch_file` and
  `load_lora_for_models`; `return_metadata` / `lora_metadata` only when the
  installed ComfyUI has them (checked by signature, not by catching
  TypeError, which would hide real errors). A module cache keeps the files of
  the last run only, so repeating a prompt doesn't reload from disk and RAM
  doesn't grow with every LoRA ever used.

## Left out

- Trigger words from LoRA metadata (`ss_tag_frequency` is noisy; sidecar
  formats differ between tools). Candidate for later, as an opt-in.
- Block weights and named arguments (`unet=`, `te=`), dual-noise strengths for
  Wan/Hunyuan.
- Semantic matching of prompt words to LoRAs (needs sentence-transformers,
  unpredictable).
- A warning when AB Wildcard's output with `<lora:` tags is wired straight into
  CLIP Text Encode (needs a graph check in the frontend).

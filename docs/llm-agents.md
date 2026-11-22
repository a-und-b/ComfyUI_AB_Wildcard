# AB Wildcard for LLM agents

Reference for an LLM that builds or runs ComfyUI workflows with this pack,
typically through a ComfyUI MCP server or the `/prompt` HTTP API. Humans: the
[README](../README.md) is the friendlier read; this file is dense on purpose.

All names below are exact API names (`class_type`, input keys, option strings).
Checked against ComfyUI 0.46.x; the nodes live in the category **AB Utils**.

---

## 1. Pick the right node

| Goal | Node (`class_type`) |
|---|---|
| Expand wildcards, `{a\|b}`, variables, conditions into a final prompt | `ABWildcardNode` |
| Let an LLM rewrite only marked parts of a prompt | `ABLLMPrep` → text generator → `ABLLMApply` |
| Write lines into a wildcard file (`__name__`) | `ABWildcardSave` |
| Read a wildcard file as text / list | `ABWildcardLoad` |
| Sort, filter, replace, dedupe, shuffle, split/join lines | `ABTextLines` |
| Add prompt pieces by chance or pick one by weight | `ABPromptMix` |

Rules of thumb:

- Randomness belongs in `ABWildcardNode` syntax, not in an LLM. LLMs pick
  "safe" entries and kill batch diversity. Use the LLM nodes to *rewrite*
  parts, not to *choose* them.
- Everything here is plain text in, plain text out. Connect the final `text`
  to `CLIPTextEncode.text` (or any STRING input).
- To read a result, wire it into `PreviewAny` (input `source`) and read the
  node's output from the history.

---

## 2. API format basics

- A link is `["<node id>", <output index>]`. Output indices are listed per node below.
- **V3 nodes** (all except `ABWildcardNode`) use `DynamicCombo` inputs. In API
  format they are **flat keys joined with `.`**: the combo's own key holds the
  chosen option, each child field is `"<combo>.<field>"`, nested combos go one
  level deeper (`"operation.split at.custom text"`). Field names may contain
  spaces, `%` and `→`; copy them exactly.
- Inputs marked *advanced* are hidden in the UI but **still required** in API
  format. Send them with their defaults.
- ComfyUI skips a node whose inputs did not change since the last run. To get
  a new random result, change a seed.
- Validation runs before execution. A file that `ABWildcardSave` creates in a
  prompt is not yet a valid `ABWildcardLoad.file` option in that same prompt;
  use it from the next prompt on, or reference it as `__name__` in
  `ABWildcardNode`, which resolves at execution time.

---

## 3. Node reference

### `ABWildcardNode` — AB Wildcard (V1 node)

| Input | Type | Notes |
|---|---|---|
| `text` | STRING | prompt with the syntax from §4 |
| `seed` | INT | same text + same seed = same output |

Outputs: `0 text` (STRING).

### `ABLLMPrep` — AB LLM Prep

| Input | Type | Values |
|---|---|---|
| `text` | STRING | prompt with `[gen …]` markers (§5) |
| `scope` | COMBO | `full` · `input` · `none` |
| `default_operation` | COMBO | `none` · `spin` · `expand` · `short` · `fit` |
| `context` | STRING, optional link | read when `scope` is `input` |

Outputs: `0 system_prompt`, `1 prompt` (STRING, to the generator), `2 plan` (`AB_LLM_PLAN`, only to `ABLLMApply`).

### `ABLLMApply` — AB LLM Apply

| Input | Type |
|---|---|
| `plan` | AB_LLM_PLAN (link from Prep output 2) |
| `generated_text` | STRING (link from the generator) |

Outputs: `0 text` (the prompt with the new parts), `1 report` (one line per part: `N op: "old" → "new"` or `N op: fallback (why), kept "old"`).

### `ABLoraLoader` — AB LoRA Loader

| Input | Type | Values |
|---|---|---|
| `model` | MODEL | link |
| `clip` | CLIP, optional link | without it the LoRAs change only the model |
| `text` | STRING | prompt with `<lora:name>`, `<lora:name:0.8>` or `<lora:name:model:clip>` tags |
| `if_missing` | COMBO | `stop with error` (default) · `skip it` |

Outputs: `0 model`, `1 clip`, `2 text` (the prompt without the tags, to CLIP Text Encode),
`3 loras` (one line per applied LoRA with its strengths). Names: any case, extension and
folder optional; the list is in `GET /ab_wildcard/wildcards` under `loras`.

### `ABWildcardSave` — AB Wildcard Save (output node)

| Input | Type | Values |
|---|---|---|
| `text` | STRING | one entry per line |
| `name` | STRING | `my/list` → `ComfyUI/wildcards/my/list.txt`, usable as `__my/list__` |
| `mode` | COMBO | `add new lines` · `add all lines` · `replace file` |

Outputs: `0 text`, `1 lines` (list output: nodes after it run once per line), `2 count` (INT). All three describe the whole file after saving.
Refused names: empty, absolute, containing `..`, starting with `~ @ !`.

### `ABWildcardLoad` — AB Wildcard Load

| Input | Type | Values |
|---|---|---|
| `file` | COMBO | lower-case keys like `examples/colors` (no `.txt`) |

Outputs: same as Save. Comments and empty lines are left out.

### `ABTextLines` — AB Text Lines

| Key | Values |
|---|---|
| `text` | STRING |
| `operation` | `sort` · `filter` · `replace` · `remove duplicates` · `shuffle` · `split and join` |
| `operation.direction` | (sort) `A → Z` · `Z → A` · `reverse order` |
| `operation.keep lines that` | (filter) `contain` · `don't contain` · `start with` · `end with` · `are exactly` · `match regex` |
| `operation.this text` | (filter) STRING; empty keeps all |
| `operation.find`, `operation.replace with` | (replace) STRING |
| `operation.use regex` | (replace) BOOLEAN; regex is case-sensitive, `\1` works |
| `operation.ignore case` | (remove duplicates) BOOLEAN |
| `operation.seed` | (shuffle) INT |
| `operation.split at`, `operation.join with` | (split and join) separator, see below |
| `input_separator`, `output_separator` | separator, advanced, required; ignored by `split and join` |

Separator values: `new line` · `comma` · `space` · `custom`; with `custom`, add `"<key>.custom text"`.
Plain matching ignores case. Lines are always trimmed, empty ones dropped.
Outputs: `0 text`, `1 lines` (list), `2 count`.

### `ABPromptMix` — AB Prompt Mix

| Key | Values |
|---|---|
| `text_1` … `text_4` | STRING, may be empty |
| `pick` | `each by chance` · `one by weight` |
| `pick.chance 1 %` … `pick.chance 4 %` | (each by chance) INT 0–100 |
| `pick.weight 1` … `pick.weight 4` | (one by weight) FLOAT 0–10 |
| `seed` | INT |
| `separator` | separator (advanced, required), default `comma` |

Outputs: `0 text`. Own RNG: does not shift `ABWildcardNode` picks.

---

## 4. Prompt syntax (`ABWildcardNode`)

```text
__colors__  <colors>        random line of wildcards/colors.txt (subfolders: __a/b__)
__2$$colors__ __1-3$$colors__  several entries
<#7$$colors>                same pick everywhere #7 is used
{red|blue|green}            one at random       {2$$a|b|c}  {1-3$$a|b|c}  {2$$ and $$a|b|c}
{5::red|3::blue|green}      weighted            {50%yes|no}  chance
{~a|b|c}                    step with the seed  {@a|b} {@x|y}  every combination  {!a|b|c}  deck
__~colors__ __@colors__ __!colors__            the same for file lines
$c = {red|blue}   $c   $c.clean .upper .lower .title     variables (assignment on its own line)
[if $c=red : hot | cold]    [if A AND NOT B : …]  AND OR NOT XOR ( )
<[mage]>  <[mage AND fire]>  <[fire OR ice]>               YAML cards by tag
__Fire Mage__               a YAML card by title
[shuffle: a, b, c]  [clean: a, , b]
// comment    # comment
```

Characters with meaning — avoid them in literal prompt text:

- `#` **anywhere** cuts the rest of the line (hex colours, `#1`, hashtags). `//` too (URLs).
  Only `__#…` and `<#…` are protected.
- `{ } |` are choices; `$name` is a variable; `__x__` and `<x>` are wildcard lookups.
- `[if …]`, `[clean: …]`, `[shuffle: …]` are commands. `[gen …]` passes through for `ABLLMPrep`.
- `<lora:name:0.8>` and other `<…>` that match no file stay literal.

Grids: `{@…}` blocks plus seeds `0 … N-1` (one queued prompt per seed) give every combination exactly once.

### Find out what exists

| Request | Returns |
|---|---|
| `GET /ab_wildcard/wildcards` | `{"wildcards": [...], "tags": [...], "globals": [...]}` — file keys and card titles, card tags, `globals.yaml` variable names |
| `GET /ab_wildcard/search?q=emerald` | `{"results": [{"token", "kind": "file"\|"card", "preview", "more"}]}`, max 50 |
| `POST /ab_wildcard/refresh` | rebuilds the index after files changed outside the graph |

Use only names from `/ab_wildcard/wildcards`. **An unknown `__name__` is not an
error: it stays literally in the output.** Check the output for leftover
`__…__` before you trust a run.

---

## 5. LLM edits (`ABLLMPrep` / `ABLLMApply`)

| Marker | Operation |
|---|---|
| `[gen: a profession fitting the character]` | fill: write what it asks for |
| `[gen spin: red silk dress]` | same meaning, other words |
| `[gen expand: castle]` | more visual detail |
| `[gen short: …]` | shorter, keeps the core |
| `[gen fit: {viking\|pirate}]` | adapt to the rest of the prompt, keep what kind of thing it is |
| `[gen do (instruction): astronaut]` | follow your instruction (required) |
| `[gen <op> (hint): …]` | any operation plus an extra instruction, e.g. `[gen expand (focus on light): castle]` |

Directions are not rewritten; every part follows them, and they are removed
from the output (with one neighbouring comma, or their whole line):

| Marker | Shown to the LLM as |
|---|---|
| `[gen theme: …]` | `- Theme: …` |
| `[gen angle: …]` | `- Camera: … (added to the prompt as it is: don't repeat it, show each part as seen from there)`; Apply appends the text to the end of the output |
| `[gen avoid: …]` | `- Avoid: …` |

Several markers of one kind are joined with `; `. Without any part to edit,
directions are just removed and the text passes through.

- Put `ABWildcardNode` **before** Prep: wildcards inside a marker resolve first
  (`[gen fit: {viking|pirate}]` → `[gen fit: viking]`), so `fit` smooths a random pick.
- Text outside markers is copied by code and never reaches the LLM.
- No markers + `default_operation` ≠ `none`: the whole text is one part.
  No markers + `none`: Prep sends "Reply with OK.", Apply passes the text through unchanged.
- `scope`: `full` the LLM sees the whole prompt with parts shown as `[N: …]` ·
  `input` it sees only `context` · `none` it sees only the parts.
  `fit` needs `full` or `input`; otherwise the report says "nothing to fit it to".
- An unknown operation (`[gen foo: …]`) raises an error in Prep.
- Don't put `#` or `//` inside a marker (comment stripping).
- Weights and `<…>` tags never reach the LLM. `([gen spin: x]:1.3)` and
  `[gen spin: (x:1.3)]` both give `(new text:1.3)`; `<lora:…>` inside a part is
  put back behind the answer. A weight on *part* of a part (`[gen expand: (gold:1.2) crown]`)
  is dropped and the report says `weight (gold:1.2) dropped`. A part that is only
  a tag (`[gen spin: <lora:x:1>]`) raises an error.
- A part keeps its original text when the answer has no line for it, the line
  is empty, or a `short` answer is longer than the original. Read `report`.

Generator: core `TextGenerateV2` ("Generate Text"), fed by `CLIPLoader`:

| Key | Value |
|---|---|
| `clip` | link from `CLIPLoader` output 0 |
| `system_prompt` | link from Prep output 0 |
| `prompt` | link from Prep output 1 |
| `max_length` | `512` |
| `sampling_mode` | `on` |
| `sampling_mode.temperature` / `.top_k` / `.top_p` / `.min_p` / `.repetition_penalty` / `.presence_penalty` / `.seed` | `0.7` / `64` / `0.95` / `0.05` / `1.05` / `0` / any |
| `thinking` | `false` |
| `use_default_template` | `true` — **required**, otherwise `system_prompt` is ignored |
| `mtp` | `auto` |

Wire `generated_text` (output 0) into Apply. Any other text generator works too
(Ollama, API nodes) as long as it takes a system prompt and a prompt.

Encoders that generate (ComfyUI 0.46.2): `qwen3.5_9b_int8_convrot.safetensors`
(type `qwen_image`, best) and `qwen_3_4b.safetensors` (type `lumina2`, faster,
weaker on `fit` and `[gen: …]`). `qwen_3_8b_fp8mixed` cannot generate.

---

## 5a. Writing instructions for the small LLM

You (the agent) write the instructions once; the local model applies them on
every queued run, to whatever the wildcards picked. That is the point of the
LLM nodes: per-run adaptation without a round trip to you. If a single fixed
prompt is all you need, write it yourself and skip Prep/Apply.

What the local model sees, for

```text
$t = {dark fairy tale|solarpunk}
[gen theme: $t]
[gen angle: low angle, 24mm lens]
a portrait of a {knight|pilot}, [gen spin (more ornate): red silk dress], [gen fit: neon street]
```

(after AB Wildcard picked `dark fairy tale` and `knight`):

```text
The prompt (parts to edit marked [N: …]):
a portrait of a knight, [1: red silk dress], [2: neon street]

Direction for every task:
- Theme: dark fairy tale
- Camera: low angle, 24mm lens (added to the prompt as it is: don't repeat it, show each part as seen from there)

Tasks:
Task 1: Say "red silk dress" differently: … Above all: more ornate. Follow the direction above.
Task 2: Check whether "neon street" belongs to the same era and world as the rest of the prompt. If it clearly does, write it unchanged. … Follow the direction above.

Answer with exactly 2 lines, "1: …", "2: …".
```

Apply then appends `, low angle, 24mm lens` to the finished prompt.

The answer format and the rules around it are fixed in code; you never write
them. You write only theme, angle, avoid, hints and `do` instructions.

Rules that work with the recommended 9B encoder:

- **Write for every pick, not for one.** `[gen fit (make the armour fit): $who]`
  breaks when `$who` is a pilot. Test with several seeds.
- **Concrete over abstract.** `muted greens, fog, wet stone` works;
  `melancholic vibe` gets ignored or overdone.
- **One change per hint.** Several wishes go into several parts or into a direction.
- **Give limits as numbers**: `max 6 words`.
- **Avoid negations in hints and themes** (`no people` tends to add people).
  Use `[gen avoid: people]`; it is phrased as a separate rule.
- **Weights belong to you, not to the local model.** Weight a whole part
  (`([gen spin: red dress]:1.3)`); never ask the model in a hint to add weights.
- **Never use `:`, `(`, `)` or `]` inside a hint**, and no `#` or `//` anywhere in a
  marker. A `:` ends the hint; `#` and `//` cut the line.
- **Content angle vs camera angle:** what to emphasise goes into a hint
  (`expand (focus on materials)`), where the camera is goes into `[gen angle: …]`.
  Theme and avoid reach the LLM only; the angle also lands in the image prompt
  (appended by Apply), so don't write the camera into the text a second time.
- **Don't let the LLM choose.** Put the options into `{a|b|c}` or a wildcard file
  and let the LLM adapt the pick; LLM choices collapse to the same safe answer.

Loop before you queue a batch:

1. Wire Prep output 1 (`prompt`) and Apply output 1 (`report`) into `PreviewAny`.
2. Queue 3–5 runs with different `ABWildcardNode.seed` and `sampling_mode.seed`.
3. Read the prompt the model got and the report: a `fallback` means no usable
   answer; an unchanged or odd answer means the instruction was unclear.
4. Sharpen the hint or direction, repeat, then queue the batch.

To change the prompt texts themselves (`LLM_TASKS` etc. in `ops.py`), run
`scripts/llm_eval.py --local-prep` from the checkout: Prep and Apply run locally,
only the generation runs on the server, so no restart is needed per round.

---

## 6. Recipes (API format)

### Prompt with an LLM pass

```json
{
  "1": {"class_type": "ABWildcardNode", "inputs": {
    "text": "a portrait of a {knight|ranger}, [gen spin: red silk dress], [gen fit: {astronaut|neon street}], oil painting",
    "seed": 1}},
  "2": {"class_type": "ABLLMPrep", "inputs": {
    "text": ["1", 0], "scope": "full", "default_operation": "none"}},
  "3": {"class_type": "CLIPLoader", "inputs": {
    "clip_name": "qwen3.5_9b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
  "4": {"class_type": "TextGenerateV2", "inputs": {
    "clip": ["3", 0], "system_prompt": ["2", 0], "prompt": ["2", 1], "max_length": 512,
    "sampling_mode": "on", "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64,
    "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05,
    "sampling_mode.repetition_penalty": 1.05, "sampling_mode.presence_penalty": 0.0,
    "sampling_mode.seed": 0, "thinking": false, "use_default_template": true, "mtp": "auto"}},
  "5": {"class_type": "ABLLMApply", "inputs": {"plan": ["2", 2], "generated_text": ["4", 0]}},
  "6": {"class_type": "PreviewAny", "inputs": {"source": ["5", 0]}},
  "7": {"class_type": "PreviewAny", "inputs": {"source": ["5", 1]}}
}
```

In an image workflow, connect `["5", 0]` to `CLIPTextEncode.text` instead of node 6.

### New wildcard entries from an LLM

Small models add intro lines ("Here are 20 professions:"). Keep only numbered
lines **before** stripping the numbers, or the intro ends up in the file.

```json
{
  "1": {"class_type": "CLIPLoader", "inputs": {
    "clip_name": "qwen3.5_9b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
  "2": {"class_type": "TextGenerateV2", "inputs": {
    "clip": ["1", 0],
    "prompt": "List 20 different professions for a medieval fantasy world. One per line, numbered, only the name of the profession.",
    "max_length": 512, "sampling_mode": "on", "sampling_mode.temperature": 0.7,
    "sampling_mode.top_k": 64, "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05,
    "sampling_mode.repetition_penalty": 1.05, "sampling_mode.presence_penalty": 0.0,
    "sampling_mode.seed": 0, "thinking": false, "use_default_template": true, "mtp": "auto"}},
  "3": {"class_type": "ABTextLines", "inputs": {
    "text": ["2", 0], "operation": "filter",
    "operation.keep lines that": "match regex", "operation.this text": "^\\s*\\d+[.:)]",
    "input_separator": "new line", "output_separator": "new line"}},
  "4": {"class_type": "ABTextLines", "inputs": {
    "text": ["3", 0], "operation": "replace",
    "operation.find": "^\\s*\\d+[.:)]\\s*", "operation.replace with": "", "operation.use regex": true,
    "input_separator": "new line", "output_separator": "new line"}},
  "5": {"class_type": "ABWildcardSave", "inputs": {
    "text": ["4", 0], "name": "llm/professions", "mode": "add new lines"}}
}
```

Change `sampling_mode.seed` for more entries. Read the file before relying on
it: small models repeat themselves.

### Several pieces by chance, then resolved

```json
{
  "1": {"class_type": "ABPromptMix", "inputs": {
    "text_1": "a cat", "text_2": "__examples/colors__ fur", "text_3": "in the rain", "text_4": "",
    "pick": "each by chance", "pick.chance 1 %": 100, "pick.chance 2 %": 50,
    "pick.chance 3 %": 30, "pick.chance 4 %": 100, "seed": 0, "separator": "comma"}},
  "2": {"class_type": "ABWildcardNode", "inputs": {"text": ["1", 0], "seed": 0}}
}
```

---

## 7. Errors and what to do

| Message / symptom | Cause | Fix |
|---|---|---|
| `__name__` or `<name>` left in the output | no such file or card | look the name up via `/ab_wildcard/wildcards` |
| Prompt cut off mid-line | `#` or `//` in the text | remove them or rephrase |
| `AB LoRA Loader: there's no LoRA "…"` | typo or file missing | use a suggested name, or `if_missing: skip it` |
| `AB LoRA Loader: "…" fits several LoRAs` | same file name in two folders | write the folder: `<lora:sdxl/detail:0.8>` |
| `AB LLM Prep: unknown operation …` | `[gen foo: …]` | use `[gen: …]`, `spin`, `expand`, `short`, `fit`, `do`, `theme`, `angle`, `avoid` |
| `AB LLM Prep: … needs an instruction` | `[gen do: …]` without `(…)` | `[gen do (what to do): …]` |
| `AB LLM Prep: … nothing left to rewrite …` | the part is only a `<…>` tag or a weight | put the tag outside the marker |
| report: `weight (…) dropped` | a weight on part of a part | weight the whole part: `([gen …]:1.2)` |
| report: `fallback (no line for it in the answer)` | model ignored the format | raise `max_length`, keep `thinking` off, try the 9B encoder |
| report: `nothing to fit it to` | `fit` with `scope` `none`, or the part is the whole prompt | use `full` with surrounding text, or `input` with `context` |
| `'NoneType' object has no attribute 'device'` in Generate Text | encoder file without LM head | use an encoder from §5 |
| `AB Wildcard Save: … can't be used as a file name` | `..`, absolute path, leading `~ @ !` | use a relative name like `my/colors` |
| `AB Wildcard Load: there's no wildcard file …` / value not in list | file missing or created in the same prompt | use `__name__` in `ABWildcardNode`, or load in the next prompt |
| `… is not a valid regular expression` | broken regex in Text Lines | fix the pattern (Python `re` syntax) |
| Same result every run | inputs unchanged → cached | change the seed |

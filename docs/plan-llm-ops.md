# Plan: surgical LLM operations in the prompt chain (AB LLM Prep / Apply)

Status: released as 1.8.0 on 2026-11-21 (`feat/llm-ops` rebased onto `main`
after PR a-und-b/ComfyUI_AB_Wildcard#9, fast-forwarded, publish run
succeeded; registry showed 1.8.0 as pending review right after the run).
Checked: pytest (75, 9 new) and syntax tests green; live on dd-machine (ComfyUI
0.46.2, frontend 1.59.6): AB Wildcard → Prep → Generate Text → Apply runs from
the API and from the frontend with `qwen_3_4b` and `qwen3.5_9b_int8_convrot`,
incl. scope input, no-slot pass-through and default operation short; the slot
chain graph and the blueprint survive save → reload with every widget value
intact and run again; the blueprint ships from the pack's `subgraphs/` folder,
shows in node search under Subgraph Blueprints, inserts with its widget values
and runs. Not checked: the Legacy (non-Vue) renderer; Holger's hand test.

Found while building:
- `TextGenerateV2` takes the system prompt only via its `system_prompt` socket,
  and only with `use_default_template` on (the default). In API format its
  DynamicCombo values are flat keys: `"sampling_mode": "on",
  "sampling_mode.temperature": 0.7, …`.
- Per-operation wording (`ops.LLM_TASKS`) after live tuning: one task line per
  slot, quoting the original. Qwen3-4B now does real spin and expand (the
  spike's weak follow-through came from terse instructions); `fill` needed
  "the concrete thing, not a description of the request", `fit` needed "keep
  what kind of thing it is (a place stays a place)". 9B follows every
  operation; 4B stays weak on `fit` and on `[gen: …]` when the context
  suggests the obvious ("a profession fitting a viking" → "Viking").
- Small models end single phrases with a period; Apply strips one when the
  original had none, or a comma-separated prompt gets "…dusk., oil painting".
- An unknown operation (`[gen foo: …]`) raises in Prep instead of passing the
  marker into the image prompt.
- Subgraph blueprint in frontend 1.59: promoted widgets become subgraph inputs
  and load positionally from `widgets_values`; the blueprint file therefore
  carries `["", "full", "none", 512, 0]` explicitly. Blueprints ship from
  `<pack>/subgraphs/*.json`; the server caches the list until restart.
- **Not ours, open:** on dd-machine (frontend 1.59.6 plus the comfyui-agent-panel
  extension) every **AB Text Lines** node loses its `text` widget on reload —
  a duplicate `operation` widget takes its place, a link into `text` lands on
  `operation`, and the graph can't be queued (`reading 'serialize'`). The same
  roundtrip is clean on the local test instance (frontend 1.58.5). The panel
  adds `operation.__cmcp_store_cleanup_1` widgets, so it is the first suspect.
  So on that machine every saved graph with Text Lines (the wildcard-entries
  example included) fails until this is fixed; a graph built fresh runs.

## Prompt review (2026-11-22)

Done, offline (pytest), not yet re-checked live:
- Tasks are labelled `Task N:`. Before, they read `N: Say "…" differently`, the
  same shape as an answer line: a model that echoed the task list got the task
  text spliced into the prompt (first line per number wins).
- A part that is the whole prompt (default operation) is no longer shown twice
  under scope full.
- `fit` names what it compares against: "the rest of the prompt" (full) or
  "the context" (input). With nothing to compare against (scope none, an empty
  context, or the part is the whole prompt) Apply's report says so.
- System prompt: "Each numbered task below" → "Each task names one numbered
  part" (the tasks are in the user message, not below the system prompt).

Resolved by the live tuning below (9B): `fit` rewriting a pick that already
fits, and `expand` turning a comma list into a sentence.

Still open (only checked on 9B so far):
- One short worked example in the system prompt (different content than any
  real prompt) for format adherence on 4B; watch for copied example content.
- `short` on one or two words always asks for 1 word; maybe skip the call and
  keep the part.
- Wildcard-entries example: an intro line from the model ("Here are 20 …")
  lands in the file. Add a Text Lines filter (match regex `^\d+[.:)]`) before
  the replace, as in the recipe in `docs/llm-agents.md`.

## Directions, hints and `do` (2026-11-22)

Decided (Holger): angles are both content (what to emphasise) and camera
(where the camera is). Target model is the 9B encoder. Goal: an agent (Claude
via MCP) writes the instructions; the local LLM applies them per run.

- `[gen theme: …]`, `[gen angle: …]`, `[gen avoid: …]`: directions for every
  part, shown as a "Direction for every task" block, each task ends with
  "Follow the direction above." Removed from the output with one neighbouring
  comma or their whole line.
- Hint per marker: `[gen <op> (hint): …]` → "Above all: hint." on that task. The
  hint sits before the first colon, so `(word:1.2)` in the content stays content.
- `[gen do (instruction): …]`: rewrite by the instruction only; without one, Prep raises.
- Format contract (`N: …` lines) stays in code; agents write only content.
- Agent guide: `docs/llm-agents.md` §5a (rules for instructions, test loop).
- Checked live on 2026-11-22: see "Live eval and tuning" below. Theme works;
  the camera never reached any part, so the angle is now appended by code.

## Weights and tags (2026-11-22)

Found: everything inside a marker went through the LLM, so `(gold:1.2)` was
lost and a `<lora:…>` tag disappeared without a word. Now kept by code, not by
asking the model:
- `<…>` tags come out of the part before Prep and go back behind the answer.
- A weight on the whole part (`[gen spin: (red dress:1.3)]`) is unwrapped for
  the LLM and wraps the answer again; `([gen …]:1.3)` around the marker always
  worked, Apply never touches it.
- A weight on a piece of a part can't follow a rewrite: Prep removes it, the
  report says `weight (…) dropped`. A fallback keeps the part exactly as written.
- A part that is only a tag raises in Prep.
- Not decided: whether weights matter at all with LLM text encoders
  (Qwen-Image, Z-Image); with CLIP they clearly do.

## Live eval and tuning (2026-11-22)

On dd-machine (ComfyUI 0.46.2, `qwen3.5_9b_int8_convrot`, temperature 0.7),
`scripts/llm_eval.py`, all 9 built-in cases × seeds 1–3. Round 0 (main after
PR a-und-b/ComfyUI_AB_Wildcard#12): format kept in 27/27 runs, no fallback;
theme visible without bleeding; weights and tags kept. Weak:
- `angle`: "low angle, 24mm" appeared in none of the 9 answers. The camera is a
  property of the whole image; a per-part rewrite has no place for it.
- `expand` on a comma list came back as prose (3/3); `fit` rewrote a pick that
  already fit ("horned iron helmet" → "weathered iron helmet", 3/3).
- Hint "more ornate" mostly ignored ("crimson velvet gown").

Decided (Holger): **Apply appends the angle text to the end of the output**;
the LLM still gets it as a direction, told not to repeat it.

Changed, then two rounds with `--local-prep` (Prep/Apply from the checkout, only
generation on the server, so no restart per round; results identical to the
server-side run for the same code and seed):
- expand: "Keep its form: a list of short comma-separated phrases stays such a list, no full sentences."
- fit: "Check whether … belongs to the same era and world as …. If it clearly
  does, write it unchanged. If anything about it is from another era or world
  (for example modern technology in a medieval scene), rewrite it …". A first
  wording ("If it already matches, write it unchanged") let the astronaut in a
  knight portrait through in 2/3 runs.
- Hint: "Also: …" → "Above all: …" ("crimson brocade gown" 2/3).
- Eval flag `sentence`: no longer fires on more commas (expand is asked for a
  list now); fires when a comma list loses items or on is/are/which/where.

Round 2: no fallback, no flag except the by-design `weights` on a part-weight;
astronaut → knight 3/3, fitting helmet unchanged 2/3 ("horned helmet" once);
expand returns comma lists; camera appended in every angle run.
Not checked: the 4B encoder with the new wording.

## Goal

Targeted LLM edits on marked parts of a prompt, as nodes in the chain: spin,
expand, shorten, interpret, fit. Text outside the marked parts never passes through
the LLM, so it cannot be mangled. Each operation reads no context, the whole
prompt, or a context the user wires in.

## Non-goals

- An own LLM client or model loader. The text generator is whatever node sits
  between Prep and Apply.
- LLM-based wildcard picking: it drifts to "safe" entries and kills batch
  diversity (reasoning in `ComfyUI_AB_Prompt_Adapter/docs/KONZEPT.md` §3.2).
- Prompt weighting and target-model style: that stays with the Prompt Adapter.

## Decided (Holger, 2026-11-21)

1. **Build on the core node `TextGenerateV2` ("Generate Text", category
   `text`).** It runs on the CLIP/text encoder already loaded for the image
   model: no Ollama server, no extra VRAM, ComfyUI does memory management.
2. **Frame it in the graph, don't call it from code.** The core node is still
   moving (it is V2 already); calling its internals breaks on updates. Any
   text generator fits in the middle: core, Simple Ollama, the Claude node.
   A single-node feel comes from a subgraph blueprint, not from code.
3. **Home: AB Wildcard.** Prep and Apply are pure text processing with no
   network, so the pack keeps its promise. V3 node API, like the other
   AB Utils nodes in `tools.py`.
4. **Answer format: numbered lines `1: replacement`.** Both formats parsed
   18/18 in the spike, so reliability did not decide it. Lines need no
   escaping: a slot with quotes (lettering such as `sign reading "OPEN"`)
   cannot break the answer. JSON is not parsed.
5. **"Interpret wildcard snippets" covers three cases, none with a new
   mechanism:**
   - snippet as instruction: `[gen: …]`.
   - smoothing random picks: the `fit` operation on a marked pick (see
     Syntax). Stays surgical; a whole-prompt coherence rewrite stays with the
     Prompt Adapter's `resolve_contradictions`.
   - generating wildcard entries: no new node. Generate Text ("give 20
     professions, one per line") → AB Text Lines (replace, regex
     `^\d+[.:)]\s*` → empty, to strip numbering) → AB Wildcard Save (add new
     lines). Delivered as an example workflow.
6. **Text without markers:** the whole text is one slot, run with Prep's
   `default operation` (none · spin · expand · short · fit; default none), so
   "shorten the whole prompt" works without a marker.
7. **Text with no slots** (no markers and default operation none): Prep sends
   a trivial request ("reply OK"), Apply ignores the answer and passes the
   text through. Blocking is no option: an ExecutionBlocker on Prep's output
   also blocks Apply and so the whole image. The run costs a few tokens on an
   encoder that is loaded anyway; documented in the README.

## Design (agreed 2026-11-21)

```
AB Wildcard ──text──► AB LLM Prep ──system_prompt──► Generate Text ──generated_text──► AB LLM Apply ──text──► CLIP Text Encode
                         │        ──prompt─────────►      ▲                               ▲
           context ┄┄┄┄┄►│                       CLIP ────┘                               │
                         └──────────────────────────────plan──────────────────────────────┘
```

**Syntax** — extends the Prompt Adapter's `[gen: …]`, which already passes
through AB Wildcard untouched (verified in KONZEPT §3.4):

| Marker | Operation |
|---|---|
| `[gen: a profession fitting the character]` | fill / interpret (as today) |
| `[gen spin: red silk dress]` | a variation with the same meaning |
| `[gen expand: castle]` | more detail |
| `[gen short: …]` | shorter, keep the core |
| `[gen fit: {viking\|pirate}]` | adapt a random pick so it fits the rest of the prompt, keep its core |

Wildcards resolve inside a marker before Prep sees it (checked 2026-11-21:
`[gen fit: {viking|pirate}]` → `[gen fit: viking]`, marker kept), so `fit`
smooths random picks without touching the rest of the prompt.

Known limit, inherited: AB Wildcard's comment stripper removes `#` and `//`
inside a marker.

**AB LLM Prep**
- in: `text`, `scope` (none · full · input), `default operation`
  (none · spin · expand · short · fit), optional `context`
- out: `system_prompt`, `prompt` (to the generator), `plan` (to Apply)
- `plan` carries the original text and the slots. Own type `AB_LLM_PLAN` so it
  cannot be wired into a text input by mistake.

**AB LLM Apply**
- in: `plan`, `generated_text`
- out: `text`, `report` (per slot: replaced · fallback, and why)
- Parses the answer, splices each slot back in; a slot with a missing or
  broken answer keeps its original content. The rest of the text is copied by
  code, never by the LLM.

## Spike results (2026-11-21, dd-machine, TextGenerateV2)

One prompt with three slots (spin, expand, short to ≤ 6 words), full context,
temperature 0.7, thinking off, seeds 1–3, each model × JSON × lines. Script:
direct `/prompt` calls (CLIPLoader → TextGenerateV2 → PreviewAny).

| Encoder (CLIP type) | Parsed | Short ≤ 8 words | Notes |
|---|---|---|---|
| `qwen_3_4b` (lumina2) | JSON 3/3 · lines 3/3 | JSON 3/3 · lines 1/3 | spin = synonym swap; expand barely expands ("under a stormy sky") |
| `qwen3.5_9b_int8_convrot` (moss_image / qwen_image) | JSON 3/3 · lines 3/3 | JSON 1/3 · lines 3/3 | real expansion ("beneath a dark, churning sky"); both CLIP types give identical output |
| `qwen_3_8b_fp8mixed` (flux2) | — | — | every run fails: `'NoneType' object has no attribute 'device'` — likely an encoder-only file without LM head; not diagnosed |

Takeaways:
- Format does not decide reliability; the word limit misses in both formats,
  depending on seed. Apply must not trust limits: `short` result longer than
  the original → fallback.
- The 4B model follows the operation weakly. Per-operation instructions need
  tuning, and the default model for spin/expand should be the 9B where it fits.
- The CLIP type only matters for loading; for generation the same weights
  give the same text.
- Not every encoder file can generate. The error is raised inside Generate
  Text, before Apply runs, so our nodes cannot catch it: document which
  encoder files work.

## Checks (when built)

- pytest: marker parsing, splicing, line-format parsing (incl. `<think>` and
  stray text around the lines), per-slot fallback on missing lines and on a
  `short` result longer than its original.
- Live on dd-machine: AB Wildcard → Prep → Generate Text → Apply runs; a saved
  graph reloads with every widget value intact (frontend 1.58 misloads values
  next to seed widgets and DynamicCombos — see `plan-wildcard-io.md`).
- Example workflows: the slot chain, and Generate Text → Text Lines →
  Wildcard Save for new wildcard entries; plus the subgraph blueprint.

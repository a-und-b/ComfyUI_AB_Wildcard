# Idea: rated wildcards — results feed back into later runs

Status: idea, captured 2026-11-21. Not planned yet. Plan it in its own session
after the UX rework in `plan-wildcard-io.md` is built (it builds on Save/Load).

## Goal

A run records the wildcard template, the expanded prompt and *which option each
choice took*, together with the result. Holger rates the result in ComfyUI.
Later runs of AB Wildcard pick well-rated options more often and poorly rated
ones less often.

## Non-goals

- Rating prompt text on its own (the image is what gets judged).
- Excluding options for good: nothing drops to zero, so new or unlucky options
  still come up now and then.
- Changing sequences (`{~…}`, `{@…}`, `{!…}`, `__~file__`): they step by seed and
  must stay deterministic.

## Decided so far (Holger, 2026-11-21)

- **Rate in ComfyUI**, not in Eagle. ComfyUI has no native rating, so this
  needs a small piece of own frontend JS on the save node — a conscious
  exception to "no own JS" in the UX rework, keep it minimal.
- **Effect: shift weights gently.** Good ratings raise an option's weight,
  bad ones lower it, with a floor above zero.

## What it takes (from the first look at the code)

1. **Record, run, rate are three steps.** Rating happens after the run, when the
   image is visible, so it cannot be a widget value of the node that already
   ran. Flow: AB Wildcard resolves and writes a trace → image is generated →
   save node stores image + template + trace → Holger rates → stats per option
   → next run reads them as weights.
2. **A trace of picks in the engine.** The expanded prompt cannot be mapped back
   to its sources (`a cat, coral fur` — which `{…}` or `__file__` gave `coral`?).
   `TagSelector` and `DynamicPromptReplacer` in `nodes.py` must log
   (source, option) per pick; AB Wildcard needs an output for it.
3. **Credit assignment is noisy.** One rating hits every pick of that image; a
   bad light drags down a good colour. Averages per option settle over many
   runs; the weight shift must stay gentle for that reason.
4. **Reproducibility changes meaning.** Same seed + same ratings = same prompt;
   new ratings change later results on purpose. Worth a switch on AB Wildcard
   ("use ratings" on/off).

## Open questions

- Storage: a JSONL ledger next to the wildcards (`ComfyUI/wildcards/.ratings/`)
  or per wildcard file? Must survive pack updates and be readable by hand.
- Rating scale: thumbs up/down or 1–5 stars?
- Where the rating UI sits: on the save node's image preview, or in the queue
  / assets panel?
- How "option" is identified when a wildcard file is edited (line text as key?).
- dd-machine runs ComfyUI remotely: the ledger lives there; fine for one user,
  what about the team?

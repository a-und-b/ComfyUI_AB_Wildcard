# Plan: autocomplete for `$variables` and `<name>`

Status: step 1 (pure functions + tests) done; steps 2 (wiring) and 3 (docs) open. Written against 1.3.1.
Format note: this repo has no `.ab/`; the plan lives here like
`plan-syntax-highlighting.md`, so any session can pick it up.

## Goal

The existing autocomplete popup (today: `__` and `<[`) also opens for variables
and for the angle form of a wildcard. Observable outcome: in the AB Wildcard
node's `text` widget, in both renderers, typing `$` lists the `globals.yaml`
names plus every `$name = …` defined anywhere in the field; accepting one
inserts `$name`. Typing `<co` lists the same wildcards as `__co` and accepting
inserts `<colors>`. `__`, `<[`, counts like `{2$$…}` and plain `a < b` behave
exactly as before.

## Facts this plan rests on

- Variable order does not matter to the engine: `store_variables`
  (`nodes.py:1169`) collects every assignment line in the text before
  `replace_variables` runs. So a variable defined further down is a valid
  suggestion, and the list is "whole text", not "above the cursor".
- `globals` already arrives with the wildcards fetch and is stored as
  `ext.globals` (`js/umi_wildcards.js:398`). No backend change.
- `<name>` is the same thing as `__name__` (README, Wildcards), so it uses the
  same `ext.wildcards` list.
- The trigger today is one regex inside a `keyup` handler
  (`js/umi_wildcards.js:465`) and cannot be tested without a browser.

## Decisions

Agreed with Holger:

1. **Suggestions come from the whole field plus `globals`**, deduplicated
   case-insensitively (follows from the engine fact above).
2. **Trigger detection moves into `ab_syntax.js` as a pure function**
   (`completionContext(textBeforeCursor)` → `{ kind, query, start }` or `null`,
   plus `variableNames(text, globals)`), so it gets node tests like the
   tokenizer. `umi_wildcards.js` keeps only the popup and the insert.
3. **`<` trigger rule.** Only `<` directly followed by a letter,
   digit or `_` opens the popup: `<co` yes, `a < b` no. `a <b` in prose would
   open it, but only if `b…` fuzzy-matches a wildcard, and Space closes it.
   `<[` stays the tag trigger.
4. **Modifiers after `$name.`** (`.clean .upper .lower .title`): a fixed list
   of four, same popup, one more `kind`.

## Not in scope

- Seeded and counted forms: `__2$$co`, `<#7$$co>`, `__~co`. The current `__`
  trigger does not cover them either; one rule for all of them is a later unit.
- Variables set by CSV wildcards or cards (the frontend cannot see them;
  same limit as the diagnostics).
- Popup look, positioning at the caret, the 50-item limit, the fuzzy matcher.
- Any change to `nodes.py` or the endpoint.

## Steps

1. **Pure functions + tests.** `completionContext` and `variableNames` in
   `js/ab_syntax.js`, cases in `tests/syntax.test.mjs`.
2. **Wiring.** Replace the regex in the `keyup` handler with
   `completionContext`; pick the source by `kind` (wildcards, tags, variables,
   modifiers) and the inserted text by `kind` (`__x__`, `<x>`, `<[x]>`, `$x`,
   `.x`). Existing `__` / `<[` behaviour unchanged.
3. **Docs.** README "Finding Wildcards (Autocomplete & Dropdowns)" and the
   step list near the end; version 1.4.0 in `pyproject.toml`, not released.

## How we'll know it works

1. `node tests/syntax.test.mjs` passes with cases that fail today because the
   function does not exist: `$co` → variable; `x {2$$` → none; `$5` → none;
   `<co` → wildcard; `a < b` → none; `<[co` → tag; `__co` → wildcard;
   `$color.cl` → modifier.
2. Same test file: `variableNames("a $late\n$late = {x}\n$Late = y", ["quality"])`
   returns `quality` and `late` once each.
3. In a running ComfyUI, legacy canvas and Nodes 2.0: `$` shows a `globals`
   name and a variable defined on a later line, accepting inserts it; `<co`
   accepting inserts `<colors>`; `__co` and `<[co` still insert as before.

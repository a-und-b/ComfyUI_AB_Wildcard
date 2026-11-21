# AB Wildcard - ComfyUI Wildcard Processor

**A powerful text processing engine for ComfyUI prompts.**

Pure **text processing** focused on: Wildcards, Variables, Conditional Logic, Tag Aggregation, Weighted Selection, and Deep Nesting.

---

## Features

* **Wildcards** — Load random entries from `.txt` and `.yaml` files
* **Dynamic Choices** — `{a|b|c}` random selection with range support
* **Sequences** — `{~…}` step, `{@…}` all combinations, `{!…}` shuffled deck (no repeats per round)
* **Variables** — Define once (`$var={...}`), reuse everywhere (`$var`)
* **Conditional Logic** — `[if condition : true | false]` with AND/OR/NOT/XOR
* **Tag Aggregation** — `<[Tag]>` to pick from YAML cards by tag
* **Card by Title** — `__Card Name__` to pull a specific YAML card's prompt
* **Global Presets** — Auto-load variables from `globals.yaml`
* **In-node Helpers** — Autocomplete (type `__`, `<name`, `<[`, `$`) + Wildcards/Tags dropdowns
* **Syntax Highlighting** — Colours, variable chips and underlined mistakes while you type

---

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/a-und-b/ComfyUI_AB_Wildcard.git
pip install -r ComfyUI_AB_Wildcard/requirements.txt
```

Restart ComfyUI.

---

## Example workflows

The workflows are AB Wildcard nodes plus ComfyUI's core **Preview as Text** node, and need no models.

**Basic** — [`ab-wildcard-example.json`](example_workflows/ab-wildcard-example.json): `{a|b}` choices.

![Basic example workflow](example_workflows/ab-wildcard-preview.png)

**Features** — [`ab-wildcard-features.json`](example_workflows/ab-wildcard-features.json): a variable, a tag pick from a YAML card (`<[warrior]>`), a conditional on the variable, and two bundled wildcard files.

![Features example workflow](example_workflows/ab-wildcard-features-preview.png)

**Cascade** — [`ab-wildcard-cascade.json`](example_workflows/ab-wildcard-cascade.json): a variable set from a tag query and fed into a second query (`$shade = <[ $test and Shade]>`), using [`color-shades.yaml`](wildcards/examples/color-shades.yaml).

![Cascade example workflow](example_workflows/ab-wildcard-cascade-preview.png)

**Sequences** — [`ab-wildcard-sequences.json`](example_workflows/ab-wildcard-sequences.json): `{~…}` step, `{~2$$…}` groups, `{@…}` combinations, `{!…}` deck and `__@file__`, with the seed widget set to `increment` (see [Sequences](#sequences-step-through-options-in-order)).

![Sequences example workflow: six runs with the seed on increment](example_workflows/ab-wildcard-sequences.gif)

1. Load a workflow file.
2. Edit the prompt text.
3. Queue it and read the expanded text in **Preview as Text**.

---

## Wiring

This node is **pure text processing** — it takes text input and outputs processed text.

| Output | Connect To |
|--------|-----------|
| `text` | CLIP Text Encode or any other text input |

---

## Finding Wildcards (Autocomplete & Dropdowns)

Two ways to insert tokens without memorizing names:

- **Autocomplete** — type `__` (or `<` directly followed by a letter, e.g. `<co`) for
  wildcards/card titles, `<[` for tags, `$` for variables (`globals.yaml` names plus every
  `$name = …` in the field, wherever it is defined) and `$name.c` (first letter after the dot) for the modifiers
  `clean`, `upper`, `lower`, `title`.
  Navigate with arrow keys, accept with **Tab/Enter or mouse click**.
- **Dropdowns** — the node's **Wildcards** and **Tags** combos insert the
  matching token (`__name__` / `<[tag]>`) on selection.

Works in both the classic and **Nodes 2.0 (Vue)** renderer. Use **Refresh
Wildcards** after adding files.

---

## Syntax Highlighting and Diagnostics

The text field colours the syntax as you type and marks mistakes.

![Highlighted prompt with an underlined unknown wildcard and an unclosed brace](example_workflows/ab-wildcard-highlighting-preview.png)

- **Colours** — comments, variables (as green chips), wildcards, tag queries, choices, sequences (`~ @ !`), counts and weights, `[if …]` and `[clean:]`.
- **Red wavy underline** — an error: an unclosed `{`, `[if`, `<[`, `[clean:` or `[shuffle:`, an unmatched `}`, or `[if` without `:`.
- **Amber wavy underline** — a warning: a wildcard or tag the server does not know, `$name` never assigned, a count larger than the option list (`{3$$a|b}`), an empty `{}`, two identical `{!…}` blocks (they draw the same option), or `{@…}` inside `{@…}` (not counted evenly).
- **Blue dotted underline** — a hint: a `//` or `#` inside a block or URL starts a comment.
- A strip at the bottom of the field gives the counts and the message at the caret.

Good to know:

- Your text is never changed; the colours are a layer behind the field.
- `$name` is only reported as unassigned in prompts without wildcards, cards or tag queries, because CSV columns and cards can set variables the field cannot see. Variables from `globals.yaml` count as assigned.
- **Refresh Wildcards** re-checks the names after you add files.

---

## Find and Replace

Click into the text field and press **Cmd/Ctrl+F**. A small bar opens on the field:

- Type to mark every match (case-insensitive, plain text — no regex); the current one is stronger, with a count like `2 / 5`.
- **Enter** / **Shift+Enter** jump to the next / previous match and scroll it into view.
- **Replace** swaps the current match, **All** swaps every match. One **Cmd/Ctrl+Z** in the field undoes the last replace.
- **Esc** closes the bar and puts the caret back in the field. Cmd/Ctrl+F outside the field is unchanged.

---

## Syntax Reference

### Wildcards

```text
__colors__              Pick random line from wildcards/colors.txt
__folder/file__         Supports subdirectories
__2$$colors__           Pick 2 random entries
__1-3$$colors__         Pick 1 to 3 random entries
<colors>                Same as __colors__
<#7$$colors>            Same pick everywhere that uses #7 (repeat it to reuse the entry)
```

### Random Choices

```text
{red|blue|green}        Pick one randomly
{2$$a|b|c|d}            Pick exactly 2
{1-3$$a|b|c|d}          Pick 1 to 3
{2$$ and $$a|b|c|d}     Pick 2, joined with " and " instead of ", "
{~a|b|c}                Sequential, see "Sequences" below
{50%yes|no}             50% chance for "yes", else "no"
{5::red|3::blue|green}  Weighted choice (red=5, blue=3, green=1)
{a|{b|{c|d}}}           Nested choices (unlimited depth)
```

### Sequences (step through options in order)

These don't pick at random: they step through the options. Set the node's
**seed widget to `increment`** and every queued run takes the next step. The
step is worked out from the seed alone, so the same seed always gives the
same prompt, and other random `{a|b}` blocks are not affected.

```text
{~a|b|c}                Step: a, b, c, a, … (option = seed mod count)
{~2$$a|b|c|d}           Step in groups: "a, b", then "c, d", then "a, b" …
{~2$$ and $$a|b|c|d}    Same, joined with " and "
{@a|b} {@x|y|z}         Combinations: every @ block counts like an odometer
                        (the first one turns fastest). 2×3 = 6 runs give
                        every pairing exactly once: a x, b x, a y, b y, a z, b z
{!a|b|c|d}              Deck: shuffled order, but every option comes up once
                        before any repeats (new shuffle each round)
{@2$$a|b|c}             Groups work with @ and ! too: a window of 2 that moves
                        with the counter (@), or 2 from the shuffled rounds (!)
__~colors__             Step through the lines of a wildcard file
__~2$$colors__          Lines of a file, two at a time
__@colors__             File joins the @ combinations
__!colors__             File lines as a shuffled deck
```

Good to know:

- Two identical `{!…}` blocks in one prompt draw the same option. Make the
  options differ, or use one block and a variable.
- `{@…}` blocks inside each other are not counted evenly; keep them side by side.
- `{~1-2$$…}` with a range picks the group size at random per run.
- A wildcard file name must not start with `~`, `@` or `!`.

Tip: to render an entire grid (e.g. 3 styles × 4 lights), use `@` blocks,
start the seed at 0 and queue 12 runs ("Batch count" 12, seed `increment`).

### Variables

```text
$color = {red|blue}     Define variable
$color                  Use variable (outputs "red" or "blue")
$color.clean            Replace _ and - with spaces
$color.upper            UPPERCASE
$color.lower            lowercase
$color.title            Title Case
```

### Conditional Logic

```text
[if red : fire theme | water theme]           If "red" in prompt
[if $var=fire : hot | cold]                   Variable equality
[if A AND B : both | not both]                AND logic
[if A OR B : either | neither]                OR logic
[if NOT dark : bright scene]                  NOT logic
[if (A OR B) AND NOT C : complex logic]       Grouping
```

### Tag Aggregation (YAML Cards)

```yaml
# In wildcards/chars.yaml
Fire Mage:
  Prompts:
    - "robed wizard, fire magic, flames"
  Tags:
    - mage
    - fire
```

```text
__Fire Mage__           Pick a prompt from the "Fire Mage" card by name
<[mage]>                Pick any entry tagged "mage"
<[mage AND fire]>       Pick entry with both tags
<[fire OR ice]>         Pick entry with either tag
```

### Utility Functions

```text
[shuffle: a, b, c, d]         Randomize order
[clean: text, , with, ,gaps]  Remove double commas/spaces
```

### Comments

```text
// This line is ignored
# This is also a comment
text here  // inline comment
```

---

## Wildcards Folder Structure

```
wildcards/
├── globals.yaml          # Auto-loaded variables
├── colors.txt            # Simple text list
├── styles.txt
├── examples/
│   ├── color-shades.yaml # Cards for the cascade example
│   ├── characters.yaml   # YAML cards with Tags
│   └── environments.yaml
└── PROMPT_TEMPLATES.txt  # Example prompts
```

### Text Files (.txt)

One entry per line:

```text
red
blue
green
```

### YAML Cards (.yaml)

```yaml
Card Name:
  Description:
    - Optional description
  Prompts:
    - "prompt text 1"
    - "prompt text 2"
  Tags:
    - tag1
    - tag2
  Prefix:
    - "prepended to prompt"
  Suffix:
    - "appended to prompt"
```

### Global Variables (globals.yaml)

```yaml
quality: masterpiece, best quality
style: cinematic lighting
```

These are available as `$quality`, `$style` in all prompts.

---

## Example Prompt

```text
// Variables for consistency
$creature = {dragon|phoenix|griffin}
$color = {red|blue|gold}

// Main prompt
__examples/quality__,
a majestic $creature with $color scales,
[if $color=red : fire breathing, flames | [if $color=blue : ice breath, frost | lightning crackling]],
__examples/lighting__,
{epic|dramatic|cinematic} composition

// Use character cards
<[warrior]> battling the $creature
```

---

## Tips

1. **Batch Generation** — Use ComfyUI's "Queue Batch" (not Latent batch size) for variations
2. **Hot Reload** — Click "Refresh Wildcards" button to reload files without restart
3. **Autocomplete** — Type `__` or `<name` (wildcards/cards), `<[` (tags), `$` (variables) or `$name.x` (modifiers); accept with Tab/Enter or click
4. **Find and replace** — Cmd/Ctrl+F in the text field
5. **Debug** — Check the console for processing errors

---

## Development

```bash
pip install -r requirements-dev.txt   # pytest (runtime needs only pyyaml)
pytest
node tests/syntax.test.mjs            # tokenizer and diagnostics (plain node, no install)
```

---

## License

MIT

---

## Credits

This node is a heavily modified fork of [UmiAI](https://github.com/Tinuva88/Comfy-UmiAI).

**Changes from original**:
- **Removed**: LLM integration, LoRA handling, resolution control, negative prompt handling, API calls
- **Added**: Weighted selection syntax (`{5::red|3::blue}`), deep nesting support, custom separators (`{2$$ and $$a|b}`)
- **Refactored**: Pure text processing focus, single output node, streamlined codebase

Many thanks to the original UmiAI developers for the foundational wildcard processing architecture.

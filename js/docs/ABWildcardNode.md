# AB Wildcard

Expands wildcards, `{a|b}` choices, variables and conditions in a prompt and
outputs plain text for any CLIP Text Encode. The same text and seed always give
the same prompt. The full guide, with examples, is in the README on GitHub.

## Parameters

- **text**: the prompt with the syntax below. Autocomplete (`Tab`), syntax
  highlighting and find/replace work in this field.
- **seed**: same text + same seed = same prompt. Set it to `increment` to step
  through `{~…}`, `{@…}` and `{!…}` blocks, one step per queued run.

## Outputs

- **text**: the expanded prompt.
- **raw**: the input exactly as typed (comments and blank lines included).
  Connect it to **AB Wildcard Extract**: `text` has its choices and lists
  already resolved, so Extract would find nothing.

## Syntax

```text
__colors__              a random line from wildcards/colors.txt
__folder/file__         subfolders work
__2$$colors__           2 random entries (__1-3$$colors__: 1 to 3)
{red|blue|green}        one at random
{5::red|3::blue|green}  weighted
{2$$a|b|c|d}            exactly 2 (joined with ", ")
{50%yes|no}             50 % chance for "yes"
{a|{b|{c|d}}}           nested
```

### Sequences

```text
{~a|b|c}                step: a, b, c, a, …
{@a|b} {@x|y|z}         every combination once (the first block turns fastest)
{!a|b|c|d}              shuffled deck: each option once before any repeats
__~colors__             step through the lines of a file (also __@…__, __!…__)
```

### Variables and conditions

```text
$hair = {red|black}     pick once, reuse with $hair
$hair.upper             also .lower .title .clean
[if $hair=red AND NOT dark : freckles | plain]
```

### Tags, LoRAs, comments

```text
<[mage AND fire]>       an entry tagged mage and fire (YAML cards)
<lora:name:0.8>         kept in the text; load it with AB LoRA Loader
[gen spin: red dress]   a part for AB LLM Prep / Apply
// comment              also # comment
```

## Tips

- A file name must not start with `~`, `@` or `!`.
- Two identical `{!…}` blocks draw the same option.
- To render a grid (3 styles × 4 lights), use `@` blocks, start the seed at 0
  and queue 12 runs with the seed on `increment`.

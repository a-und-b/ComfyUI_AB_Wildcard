"""AB Utils nodes: save/load wildcard files, line tools, prompt mixing, LLM prep/apply,
LoRA loader (V3 node API).

Thin wrappers: the logic lives in ops.py and loras.py, where pytest can reach it.
Plans: docs/plan-wildcard-io.md, docs/plan-llm-ops.md, docs/plan-lora-loader.md
"""
import logging
import os

import folder_paths
from comfy_api.latest import io

from . import loras, ops
from .nodes import TagLoader, get_all_wildcard_paths

CATEGORY = "AB Utils"
UINT64 = 0xffffffffffffffff


def _wildcards_dir():
    return os.path.join(folder_paths.base_path, "wildcards")


def _txt_files():
    """Wildcard .txt files across all wildcard folders: lower-cased key -> path."""
    return TagLoader(get_all_wildcard_paths(), {}).txt_lookup


def _separator(id, label, first="new line", advanced=False):
    """Separator dropdown; a text field appears only for 'custom'. `first` is the default.

    Fields inside a DynamicCombo show their id, not display_name (frontend 1.58), so
    ids there are the readable labels.
    """
    names = [first] + [s for s in ops.SEPARATORS if s != first]
    return io.DynamicCombo.Input(id, display_name=label, options=[
        io.DynamicCombo.Option(n, [io.String.Input("custom text")] if n == "custom" else [])
        for n in names
    ], extra_dict={"advanced": True} if advanced else None)  # DynamicCombo.Input takes no `advanced` arg


def _sep(value):
    """DynamicCombo value -> (choice, custom) for ops.split / ops.join."""
    choice = next(v for k, v in value.items() if k != "custom text")
    return choice, value.get("custom text", "")


def _lines_out(items):
    return io.NodeOutput("\n".join(items), items, len(items))


LINES_OUTPUTS = [
    io.String.Output(display_name="text", tooltip="All lines as one text"),
    io.String.Output(display_name="lines", tooltip="One item per line: nodes after this run once per line",
                     is_output_list=True),
    io.Int.Output(display_name="count"),
]


class ABWildcardSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ABWildcardSave",
            display_name="AB Wildcard Save",
            category=CATEGORY,
            description="Saves lines as a wildcard file in ComfyUI/wildcards and outputs the whole file.",
            search_aliases=["save wildcard", "write wildcard file", "save list", "save lines"],
            is_output_node=True,
            inputs=[
                io.String.Input("text", multiline=True, tooltip="One wildcard entry per line. Type it or connect a text."),
                io.String.Input("name", default="my/list",
                                tooltip="Saved as ComfyUI/wildcards/<name>.txt and usable as __<name>__"),
                io.Combo.Input("mode", options=ops.SAVE_MODES, tooltip=(
                    "add new lines: only lines the file doesn't have yet · "
                    "add all lines: append everything · replace file: start over")),
            ],
            outputs=LINES_OUTPUTS,
        )

    @classmethod
    def execute(cls, text, name, mode):
        _key, path = ops.safe_path(_wildcards_dir(), name)
        items = ops.save_lines(path, text, mode)
        ops.clear_wildcard_caches()
        return _lines_out(items)


class ABWildcardLoad(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ABWildcardLoad",
            display_name="AB Wildcard Load",
            category=CATEGORY,
            description="Reads a wildcard file: all lines as text, as a list, and how many.",
            search_aliases=["load wildcard", "read wildcard file", "load list", "read lines"],
            inputs=[
                io.Combo.Input("file", options=sorted(_txt_files()) or [""],
                               tooltip="New files appear here after reloading the browser page"),
            ],
            outputs=LINES_OUTPUTS,
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        # ponytail: always re-read; a small text file is cheaper than tracking mtimes
        return float("nan")

    @classmethod
    def execute(cls, file):
        path = _txt_files().get(ops.file_key(file).lower())
        if not path:
            raise ValueError(f"AB Wildcard Load: there's no wildcard file {file!r}. Reload the page to refresh the list.")
        return _lines_out(ops.load_lines(path))


class ABTextLines(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        Opt, S = io.DynamicCombo.Option, io.String.Input
        return io.Schema(
            node_id="ABTextLines",
            display_name="AB Text Lines",
            category=CATEGORY,
            description="One job on a list of lines: sort, filter, replace, remove duplicates, shuffle, "
                        "or split and join. Chain several nodes for several jobs.",
            search_aliases=["sort lines", "filter lines", "replace text", "remove duplicates", "dedupe",
                            "shuffle lines", "split text", "join text", "list"],
            inputs=[
                S("text", multiline=True, tooltip="One item per line (other separators under advanced)"),
                io.DynamicCombo.Input("operation", options=[
                    Opt("sort", [io.Combo.Input("direction", options=ops.SORTS)]),
                    Opt("filter", [
                        io.Combo.Input("keep lines that", options=ops.MATCHES),
                        # Not "text": a child id equal to a top-level input name makes comfyui-agent-panel
                        # delete the top-level widget on load (it takes it for a stale DynamicCombo row).
                        S("this text", tooltip="Empty keeps every line"),
                    ]),
                    Opt("replace", [
                        S("find", tooltip="Any case; empty changes nothing"),
                        S("replace with"),
                        io.Boolean.Input("use regex", default=False, advanced=True,
                                         tooltip="find is a regular expression (case-sensitive); replace with may use \\1"),
                    ]),
                    Opt("remove duplicates", [
                        io.Boolean.Input("ignore case", default=False, advanced=True),
                    ]),
                    Opt("shuffle", [io.Int.Input("seed", default=0, min=0, max=UINT64, control_after_generate=True)]),
                    Opt("split and join", [
                        _separator("split at", "split at", "comma"),
                        _separator("join with", "join with", "new line"),
                    ]),
                ]),
                _separator("input_separator", "lines in text split at", advanced=True),
                _separator("output_separator", "lines in output joined with", advanced=True),
            ],
            outputs=[
                io.String.Output(display_name="text", tooltip="The lines, joined"),
                *LINES_OUTPUTS[1:],
            ],
        )

    @classmethod
    def execute(cls, text, operation, input_separator, output_separator):
        op = operation["operation"]
        if op == "split and join":
            input_separator, output_separator = operation["split at"], operation["join with"]
        items = ops.split(text, *_sep(input_separator))
        if op == "sort":
            items = ops.sort_lines(items, operation["direction"])
        elif op == "filter":
            items = ops.filter_lines(items, operation["keep lines that"], operation["this text"])
        elif op == "replace":
            items = ops.replace_lines(items, operation["find"], operation["replace with"], operation["use regex"])
        elif op == "remove duplicates":
            items = ops.remove_duplicates(items, operation["ignore case"])
        elif op == "shuffle":
            items = ops.shuffle_lines(items, operation["seed"])
        return io.NodeOutput(ops.join(items, *_sep(output_separator)), items, len(items))


MIX_SLOTS = 4  # ponytail: fixed; growing slots with a weight each need DynamicSlot (frontend 1.58 lacks it)
SOME, ONE = "each by chance", "one by weight"


class ABPromptMix(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        slots = range(1, MIX_SLOTS + 1)
        return io.Schema(
            node_id="ABPromptMix",
            display_name="AB Prompt Mix",
            category=CATEGORY,
            description="Lets prompt pieces through by chance: each piece with its own chance, "
                        "or exactly one piece by weight. Same seed, same result.",
            search_aliases=["random prompt", "chance", "probability", "weighted choice", "prompt pieces"],
            inputs=[
                # Widgets, not force_input sockets: sockets before a DynamicCombo and a seed make
                # frontend 1.58 drop saved values on load. Linking a text still works.
                *[io.String.Input(f"text_{i}", display_name=f"piece {i}", default="",
                                  tooltip="Type a prompt piece or connect a text") for i in slots],
                io.DynamicCombo.Input("pick", options=[
                    io.DynamicCombo.Option(SOME, [
                        io.Int.Input(f"chance {i} %", default=100, min=0, max=100,
                                     display_mode=io.NumberDisplay.slider) for i in slots]),
                    io.DynamicCombo.Option(ONE, [
                        io.Float.Input(f"weight {i}", default=1.0, min=0.0, max=10.0,
                                       step=0.1) for i in slots]),
                ]),
                io.Int.Input("seed", default=0, min=0, max=UINT64, control_after_generate=True),
                _separator("separator", "join with", "comma", advanced=True),
            ],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def execute(cls, pick, seed, separator, **pieces):
        texts = [(pieces.get(f"text_{i}") or "").strip() for i in range(1, MIX_SLOTS + 1)]
        if pick["pick"] == SOME:
            used = ops.mix(texts, chances=[pick[f"chance {i} %"] for i in range(1, MIX_SLOTS + 1)], seed=seed)
        else:
            used = ops.mix(texts, weights=[pick[f"weight {i}"] for i in range(1, MIX_SLOTS + 1)], seed=seed)
        return io.NodeOutput(ops.join(used, *_sep(separator)))


LLMPlan = io.Custom("AB_LLM_PLAN")  # own type: can't be wired into a text input by mistake


class ABLLMPrep(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ABLLMPrep",
            display_name="AB LLM Prep",
            category=CATEGORY,
            description="Turns the [gen …] parts of a prompt into one request for a text generator "
                        "(e.g. Generate Text). AB LLM Apply puts the answers back; the rest of the "
                        "prompt never passes through the LLM.",
            search_aliases=["llm", "gen", "spin prompt", "expand prompt", "shorten prompt", "rewrite prompt",
                            "generate text", "prompt slots"],
            inputs=[
                io.String.Input("text", multiline=True, tooltip=(
                    "The prompt. Mark parts: [gen: what to write] · [gen spin: …] same meaning, other words · "
                    "[gen expand: …] more detail · [gen short: …] shorter · [gen fit: …] adapt to the rest · "
                    "[gen do (instruction): …] your own instruction. Any part takes a hint: [gen spin (more gothic): …]. "
                    "For every part: [gen theme: …] · [gen angle: camera] · [gen avoid: …] (removed from the output)")),
                io.Combo.Input("scope", options=ops.LLM_SCOPES, tooltip=(
                    "What the LLM reads besides the parts: full = the whole prompt · "
                    "input = only the context input · none = just the parts")),
                io.Combo.Input("default_operation", display_name="default operation", options=ops.LLM_DEFAULT_OPS, tooltip=(
                    "For a text without [gen …] parts: run this on the whole text. "
                    "none passes it through unchanged")),
                io.String.Input("context", force_input=True, optional=True,
                                tooltip="Read by the LLM when scope is input"),
            ],
            outputs=[
                io.String.Output(display_name="system_prompt", tooltip="To Generate Text's system_prompt"),
                io.String.Output(display_name="prompt", tooltip="To Generate Text's prompt"),
                LLMPlan.Output(display_name="plan", tooltip="To AB LLM Apply"),
            ],
        )

    @classmethod
    def execute(cls, text, scope, default_operation, context=""):
        return io.NodeOutput(*ops.llm_prep(text, scope, default_operation, context))


class ABLLMApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ABLLMApply",
            display_name="AB LLM Apply",
            category=CATEGORY,
            description="Puts a text generator's answers back into the prompt from AB LLM Prep. "
                        "A part without a usable answer keeps its original text.",
            search_aliases=["llm", "gen", "apply llm answer", "prompt slots"],
            inputs=[
                LLMPlan.Input("plan", tooltip="From AB LLM Prep"),
                io.String.Input("generated_text", force_input=True, tooltip="From Generate Text"),
            ],
            outputs=[
                io.String.Output(display_name="text", tooltip="The prompt with the new parts"),
                io.String.Output(display_name="report", tooltip="Per part: the new text, or why it kept the original"),
            ],
        )

    @classmethod
    def execute(cls, plan, generated_text):
        return io.NodeOutput(*ops.llm_apply(plan, generated_text))


class ABLoraLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ABLoraLoader",
            display_name="AB LoRA Loader",
            category=CATEGORY,
            description="Loads the LoRAs named in the prompt as <lora:name:strength> and outputs the "
                        "prompt without them. Put it between AB Wildcard and CLIP Text Encode.",
            search_aliases=["lora", "lora tag", "load lora", "lora from prompt", "prompt lora", "<lora:",
                            "a1111 lora", "lora loader"],
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip", optional=True,
                              tooltip="Optional: without it, LoRAs change only the model"),
                io.String.Input("text", multiline=True, tooltip=(
                    "The prompt, usually from AB Wildcard. <lora:name> = strength 1 · "
                    "<lora:name:0.8> = model and clip 0.8 · <lora:name:0.8:0.5> = model 0.8, clip 0.5")),
                io.Combo.Input("if_missing", display_name="if a LoRA is missing", options=loras.MISSING_MODES,
                               default=loras.MISSING_STOP, advanced=True, tooltip=(
                                   "stop with error: the run stops and names the LoRA · "
                                   "skip it: the tag is removed, the run goes on, the console warns")),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
                io.String.Output(display_name="text", tooltip="The prompt without <lora:…> tags: to CLIP Text Encode"),
                io.String.Output(display_name="loras", tooltip="Each applied LoRA with its strengths, one per line"),
            ],
        )

    @classmethod
    def execute(cls, model, text, if_missing=loras.MISSING_STOP, clip=None):
        clean, tags = loras.extract(text or "")
        files = folder_paths.get_filename_list("loras")
        picks = []
        for t in tags:
            try:
                picks.append((loras.resolve(t.name, files), t.model, t.clip))
            except LookupError as e:
                if if_missing == loras.MISSING_STOP:
                    raise ValueError(f"AB LoRA Loader: {e}") from None
                logging.warning(f"[AB LoRA Loader] skipped {t.raw}: {e}")

        applied, cache = [], {}
        for f, strength_model, strength_clip in loras.merge(picks):
            if clip is None:
                strength_clip = 0.0
                if not strength_model:
                    continue
            path = folder_paths.get_full_path_or_raise("loras", f)
            lora, meta = _load_lora_file(path)
            cache[path] = (lora, meta)
            model, clip = _apply_lora(model, clip, lora, meta, strength_model, strength_clip)
            applied.append((loras.insert_name(f, files), strength_model, strength_clip if clip is not None else None))
        _LORA_CACHE.clear()
        _LORA_CACHE.update(cache)  # keep only what this run used
        return io.NodeOutput(model, clip, clean, loras.describe(applied))


_LORA_CACHE = {}  # path -> (state dict, metadata), the LoRAs of the last run


def _load_lora_file(path):
    hit = _LORA_CACHE.get(path)
    if hit is not None:
        return hit
    import comfy.utils
    if _accepts(comfy.utils.load_torch_file, "return_metadata"):
        return comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    return comfy.utils.load_torch_file(path, safe_load=True), None  # older ComfyUI


def _apply_lora(model, clip, lora, meta, strength_model, strength_clip):
    import comfy.sd
    extra = {"lora_metadata": meta} if _accepts(comfy.sd.load_lora_for_models, "lora_metadata") else {}
    return comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip, **extra)


def _accepts(fn, arg):
    import inspect
    try:
        return arg in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False

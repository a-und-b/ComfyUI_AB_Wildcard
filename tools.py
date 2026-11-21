"""Graph-side helpers: save/load wildcard files, line tools, prompt mixing.

Plan: docs/plan-wildcard-io.md
"""
import io
import os
import random
import re

import folder_paths

try:  # inside ComfyUI this is a package; top-level "nodes" is ComfyUI's own
    from .nodes import GLOBAL_CACHE, GLOBAL_INDEX, TagLoader, get_all_wildcard_paths, read_file_lines
except ImportError:  # tests import the pack's modules top-level
    from nodes import GLOBAL_CACHE, GLOBAL_INDEX, TagLoader, get_all_wildcard_paths, read_file_lines

CATEGORY = "AB Utils"
COLOR = "#322947"


def _lines(text):
    """Non-empty, trimmed lines."""
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _key(name):
    """'My\\Colors.txt ' -> 'My/Colors' (the key scheme TagLoader uses, before lower-casing)."""
    name = (name or "").strip().replace("\\", "/")
    return name[:-4] if name.lower().endswith(".txt") else name


def _wildcards_dir():
    # Read at call time: tests point base_path elsewhere.
    return os.path.join(folder_paths.base_path, "wildcards")


def _safe_path(name):
    """Path of <ComfyUI>/wildcards/<name>.txt; refuses anything that leaves that folder."""
    key = _key(name)
    base = os.path.realpath(_wildcards_dir())
    path = os.path.realpath(os.path.join(base, key + ".txt"))
    bad = (
        not key
        or os.path.isabs(key)
        or ".." in key.split("/")
        or key[0] in "~@!"  # would read as a sequence token: __~name__
        or os.path.commonpath([base, path]) != base
    )
    if bad:
        raise ValueError(
            f"AB Wildcard Save: invalid name {name!r}. Use a relative name like 'my/colors' "
            "(no '..', not starting with ~ @ !)."
        )
    return key, path


def _clear_caches():
    """Same reset as POST /ab_wildcard/refresh, so __name__ sees the new file at once."""
    GLOBAL_CACHE.clear()
    GLOBAL_INDEX.update(built=False, files=set(), entries={})


class ABWildcardSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "tooltip": "One wildcard entry per line"}),
                "name": ("STRING", {"default": "my/list", "tooltip": "Saved as ComfyUI/wildcards/<name>.txt, used as __<name>__"}),
                "mode": (["overwrite", "append", "append unique"], {
                    "tooltip": "append unique skips lines already in the file and repeats in the input",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "name")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    COLOR = COLOR
    DESCRIPTION = "Writes lines to a wildcard file in ComfyUI/wildcards."

    def save(self, text, name, mode):
        key, path = _safe_path(name)
        existing = ""
        if mode != "overwrite" and os.path.exists(path):
            with open(path, encoding="utf8") as f:
                existing = f.read()
        new = _lines(text)
        if mode == "append unique":
            seen = set(read_file_lines(io.StringIO(existing)))
            new = [ln for ln in dict.fromkeys(new) if ln not in seen]
        if existing and not existing.endswith("\n"):
            existing += "\n"

        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"  # write then rename: a crash never leaves half a file
        with open(tmp, "w", encoding="utf8") as f:
            f.write(existing + "".join(ln + "\n" for ln in new))
        os.replace(tmp, path)
        _clear_caches()
        return (text, key)


def _txt_files():
    """Wildcard .txt files across all wildcard folders: lower-cased key -> path."""
    return TagLoader(get_all_wildcard_paths(), {}).txt_lookup


class ABWildcardLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": (sorted(_txt_files()) or [""], {"tooltip": "Wildcard .txt file"}),
            },
            "optional": {
                "name": ("STRING", {"forceInput": True, "tooltip": "Overrides file, e.g. from AB Wildcard Save"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("text", "lines", "count")
    OUTPUT_IS_LIST = (False, True, False)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    COLOR = COLOR
    DESCRIPTION = "Reads a wildcard file: whole text, one list item per line, line count."

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # ponytail: always re-read; a small text file is cheaper than tracking mtimes
        # through linked inputs (Save -> Load in one run must see the new lines).
        return float("nan")

    def load(self, file, name=None):
        key = _key(name or file)
        path = _txt_files().get(key.lower())
        if not path:
            raise ValueError(f"AB Wildcard Load: no wildcard file {key!r}.")
        with open(path, encoding="utf8") as f:
            lines = read_file_lines(f)
        return ("\n".join(lines), lines, len(lines))


def _unescape(sep):
    """Widgets can't hold a real newline in a one-line field: accept \\n and \\t."""
    return (sep or "").replace("\\n", "\n").replace("\\t", "\t")


def _compile(pattern, regex, field):
    """Pattern -> compiled regex. Plain text matches literally and case-insensitively."""
    if not regex:
        return re.compile(re.escape(pattern), re.IGNORECASE)
    try:
        return re.compile(pattern)
    except re.error as e:
        raise ValueError(f"AB Text Lines: invalid regex in '{field}': {e}") from None


ORDERS = ["as is", "sort", "sort desc", "reverse", "shuffle"]


def text_lines(text, split_by="\\n", find="", replace="", keep="", drop="",
               regex=False, dedupe=True, order="as is", seed=0, join_with="\\n"):
    """The AB Text Lines pipeline; every step is a no-op at its default.

    split -> find/replace -> trim, drop empty -> keep -> drop -> dedupe -> order -> join
    """
    sep = _unescape(split_by)
    items = text.split(sep) if sep else [text]
    if find:
        rx = _compile(find, regex, "find")
        # plain mode: the replacement is literal too (no \1 or \\ handling)
        repl = replace if regex else (lambda _m: replace)
        items = [rx.sub(repl, it) for it in items]
    items = [it.strip() for it in items if it.strip()]
    if keep:
        rx = _compile(keep, regex, "keep")
        items = [it for it in items if rx.search(it)]
    if drop:
        rx = _compile(drop, regex, "drop")
        items = [it for it in items if not rx.search(it)]
    if dedupe:
        items = list(dict.fromkeys(items))
    if order == "sort":
        items.sort(key=str.casefold)
    elif order == "sort desc":
        items.sort(key=str.casefold, reverse=True)
    elif order == "reverse":
        items.reverse()
    elif order == "shuffle":
        random.Random(seed).shuffle(items)
    return _unescape(join_with).join(items), items


class ABTextLines:
    @classmethod
    def INPUT_TYPES(cls):
        s = lambda default, tip: ("STRING", {"default": default, "tooltip": tip})
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "split_by": s("\\n", "Split into items here (\\n = newline, \\t = tab, empty = one item)"),
                "find": s("", "Replace this in every item (empty = off)"),
                "replace": s("", "Replacement for find; with regex, \\1 inserts a group"),
                "keep": s("", "Keep only items containing this (empty = all)"),
                "drop": s("", "Remove items containing this (empty = none)"),
                "regex": ("BOOLEAN", {"default": False, "tooltip": "find/keep/drop are regular expressions (case-sensitive); off = plain text, any case"}),
                "dedupe": ("BOOLEAN", {"default": True, "tooltip": "Remove repeated items, first one wins"}),
                "order": (ORDERS,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "For shuffle"}),
                "join_with": s("\\n", "Put between items in the text output"),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("text", "lines", "count")
    OUTPUT_IS_LIST = (False, True, False)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    COLOR = COLOR
    DESCRIPTION = (
        "Split, replace, trim, filter, dedupe, sort or shuffle, join — in that order. "
        "Items are always trimmed and empty ones dropped."
    )

    def run(self, text, **kwargs):
        joined, items = text_lines(text, **kwargs)
        return (joined, items, len(items))


MIX_SLOTS = 4  # ponytail: fixed slots; growing inputs need frontend JS, add when 4 is too few


class ABPromptMix:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, MIX_SLOTS + 1):
            optional[f"text_{i}"] = ("STRING", {"forceInput": True})
            optional[f"weight_{i}"] = ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "each by chance: probability this piece is used. pick one weighted: its relative weight",
            })
        return {
            "required": {
                "mode": (["each by chance", "pick one weighted"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "separator": ("STRING", {"default": ", ", "tooltip": "Between used pieces (\\n = newline)"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "mix"
    CATEGORY = CATEGORY
    COLOR = COLOR
    DESCRIPTION = (
        "Lets prompt pieces through by chance: each piece with its own probability, "
        "or exactly one piece by weight. Same seed, same result."
    )

    def mix(self, mode, seed, separator, **slots):
        rng = random.Random(seed)  # own RNG: the global one is seeded by AB Wildcard
        pieces = [
            ((slots.get(f"text_{i}") or "").strip(), float(slots.get(f"weight_{i}", 1.0)))
            for i in range(1, MIX_SLOTS + 1)
        ]
        if mode == "each by chance":
            # one roll per slot, connected or not: wiring slot 3 never changes slot 1's luck
            rolls = [rng.random() for _ in pieces]
            used = [t for (t, w), r in zip(pieces, rolls) if t and r < w]
        else:
            live = [(t, w) for t, w in pieces if t and w > 0]
            used = rng.choices([t for t, _ in live], weights=[w for _, w in live]) if live else []
        return (_unescape(separator).join(used),)

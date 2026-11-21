"""Graph-side helpers: save/load wildcard files, line tools, prompt mixing.

Plan: docs/plan-wildcard-io.md
"""
import io
import os

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

"""Pure logic behind the AB Utils nodes in tools.py.

No comfy_api import here, so pytest reaches all of it without ComfyUI.
Plan: docs/plan-wildcard-io.md
"""
import io
import os
import random
import re

try:  # inside ComfyUI this is a package; top-level "nodes" is ComfyUI's own
    from .nodes import GLOBAL_CACHE, GLOBAL_INDEX, read_file_lines
except ImportError:  # tests import the pack's modules top-level
    from nodes import GLOBAL_CACHE, GLOBAL_INDEX, read_file_lines


def tidy(items):
    """Trimmed, non-empty items."""
    return [it.strip() for it in items if it and it.strip()]


# --- separators (labels shown in the UI) ---------------------------------------

SEPARATORS = ["new line", "comma", "space", "custom"]
_SPLIT = {"new line": None, "comma": ",", "space": " "}   # None: str.splitlines (also \r\n)
_JOIN = {"new line": "\n", "comma": ", ", "space": " "}  # comma joins the prompt way: ", "


def split(text, choice="new line", custom=""):
    sep = custom if choice == "custom" else _SPLIT[choice]
    if sep is None:
        return tidy((text or "").splitlines())
    return tidy((text or "").split(sep)) if sep else tidy([text])


def join(items, choice="new line", custom=""):
    return (custom if choice == "custom" else _JOIN[choice]).join(items)


# --- line operations ------------------------------------------------------------

def compile_regex(pattern, what):
    try:
        return re.compile(pattern)
    except re.error as e:
        raise ValueError(f"AB Text Lines: {what} is not a valid regular expression: {e}") from None


SORTS = ["A → Z", "Z → A", "reverse order"]


def sort_lines(items, direction="A → Z"):
    if direction == "reverse order":
        return items[::-1]
    return sorted(items, key=str.casefold, reverse=direction == "Z → A")


MATCHES = ["contain", "don't contain", "start with", "end with", "are exactly", "match regex"]


def filter_lines(items, how="contain", text=""):
    """Plain matches ignore case; an empty text keeps everything."""
    if not text:
        return items
    if how == "match regex":
        rx = compile_regex(text, "text")
        return [it for it in items if rx.search(it)]
    t = text.casefold()
    test = {
        "contain": lambda s: t in s,
        "don't contain": lambda s: t not in s,
        "start with": lambda s: s.startswith(t),
        "end with": lambda s: s.endswith(t),
        "are exactly": lambda s: s == t,
    }[how]
    return [it for it in items if test(it.casefold())]


def replace_lines(items, find="", replace_with="", regex=False):
    """Plain: literal and any case. Regex: case-sensitive, replace_with may use \\1."""
    if not find:
        return items
    if regex:
        rx, repl = compile_regex(find, "find"), replace_with
    else:
        rx, repl = re.compile(re.escape(find), re.IGNORECASE), (lambda _m: replace_with)
    return tidy(rx.sub(repl, it) for it in items)


def remove_duplicates(items, ignore_case=False):
    seen, out = set(), []
    for it in items:
        k = it.casefold() if ignore_case else it
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def shuffle_lines(items, seed=0):
    items = list(items)
    random.Random(seed).shuffle(items)
    return items


# --- prompt mix -------------------------------------------------------------------

def mix(pieces, chances=None, weights=None, seed=0):
    """Pick prompt pieces. Give chances (0–100 %, each piece on its own) or weights (one piece).

    Empty pieces are never used. One roll per slot, wired or not, so wiring
    slot 3 never changes slot 1's luck. Own RNG: AB Wildcard seeds the global one.
    """
    rng = random.Random(seed)
    if chances is not None:
        rolls = [rng.random() * 100 for _ in pieces]
        return [p for p, c, r in zip(pieces, chances, rolls) if p and r < c]
    live = [(p, w) for p, w in zip(pieces, weights) if p and w > 0]
    return rng.choices([p for p, _ in live], weights=[w for _, w in live]) if live else []


# --- wildcard files ----------------------------------------------------------------

def file_key(name):
    """'My\\Colors.txt ' -> 'My/Colors' (TagLoader's key scheme, before lower-casing)."""
    name = (name or "").strip().replace("\\", "/")
    return name[:-4] if name.lower().endswith(".txt") else name


def safe_path(base, name):
    """(key, path) of <base>/<name>.txt; refuses anything that leaves base."""
    key = file_key(name)
    base = os.path.realpath(base)
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
            f"AB Wildcard Save: {name!r} can't be used as a file name. "
            "Use a relative name like my/colors (no '..', not starting with ~ @ !)."
        )
    return key, path


SAVE_MODES = ["add new lines", "add all lines", "replace file"]


def save_lines(path, text, mode="add new lines"):
    """Write lines to path; returns every line of the file afterwards (as Load would read it)."""
    existing = ""
    if mode != "replace file" and os.path.exists(path):
        with open(path, encoding="utf8") as f:
            existing = f.read()
    new = tidy((text or "").splitlines())
    if mode == "add new lines":
        seen = set(read_file_lines(io.StringIO(existing)))
        new = [ln for ln in dict.fromkeys(new) if ln not in seen]
    if existing and not existing.endswith("\n"):
        existing += "\n"
    content = existing + "".join(ln + "\n" for ln in new)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"  # write then rename: a crash never leaves half a file
    with open(tmp, "w", encoding="utf8") as f:
        f.write(content)
    os.replace(tmp, path)
    return read_file_lines(io.StringIO(content))


def clear_wildcard_caches():
    """Same reset as POST /ab_wildcard/refresh, so __name__ sees a saved file at once."""
    GLOBAL_CACHE.clear()
    GLOBAL_INDEX.update(built=False, files=set(), entries={})


def load_lines(path):
    with open(path, encoding="utf8") as f:
        return read_file_lines(f)

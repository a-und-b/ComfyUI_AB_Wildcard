"""<lora:name:weight> tags: parse, resolve to files, strip from the prompt.

Pure logic for the AB LoRA Loader node (tools.py), reachable from pytest.
The name rules here are mirrored in js/ab_syntax.js (loraMatches,
loraInsertNames) for autocomplete and the "unknown LoRA" warning.
Plan: docs/plan-lora-loader.md
"""
import difflib
import os
import re
from dataclasses import dataclass

# Same set as ComfyUI's folder_paths.supported_pt_extensions.
EXTENSIONS = ('.safetensors', '.ckpt', '.pt', '.pt2', '.bin', '.pth', '.pkl', '.sft')

# <lora:…> and A1111's older <lyco:…>; any case; one line; no nested <>.
TAG_RE = re.compile(r"<(?:lora|lyco):([^<>\n]*)>", re.IGNORECASE)
_NUM_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")

MISSING_STOP, MISSING_SKIP = "stop with error", "skip it"
MISSING_MODES = [MISSING_STOP, MISSING_SKIP]


@dataclass
class LoraTag:
    name: str
    model: float
    clip: float
    raw: str


def parse_spec(spec):
    """'name', 'name:0.8' or 'name:0.8:0.5' -> (name, model, clip).

    Numbers are taken from the right, so a name may contain colons. One number
    sets both strengths, two set model and clip. Raises ValueError without a name.
    """
    parts = [p.strip() for p in spec.split(":")]
    nums = []
    while len(parts) > 1 and len(nums) < 2 and _NUM_RE.fullmatch(parts[-1]):
        nums.insert(0, float(parts.pop()))
    name = ":".join(parts).strip()
    if not name:
        raise ValueError("has no name")
    model = nums[0] if nums else 1.0
    clip = nums[1] if len(nums) > 1 else model
    return name, model, clip


def extract(text):
    """Text -> (text without tags, [LoraTag]) in order of appearance.

    Only lines that held a tag are tidied (doubled commas, spaces, a comma left
    at either end); a line that held nothing but tags goes away. Every other
    line stays exactly as it was.
    """
    tags, bad = [], []
    out_lines = []
    for line in text.split("\n"):
        found = list(TAG_RE.finditer(line))
        if not found:
            out_lines.append(line)
            continue
        for m in found:
            try:
                tags.append(LoraTag(*parse_spec(m.group(1)), raw=m.group(0)))
            except ValueError as e:
                bad.append(f"{m.group(0)} {e}")
        rest = TAG_RE.sub(" ", line)
        if not rest.strip(" \t,"):
            continue
        rest = re.sub(r"\s*,(?:\s*,)+", ",", rest)   # ", ," -> ","
        rest = re.sub(r"[ \t]+,", ",", rest)          # "a ," -> "a,"
        rest = re.sub(r"[ \t]{2,}", " ", rest)
        out_lines.append(rest.strip(" \t,"))
    if bad:
        raise ValueError("AB LoRA Loader: " + "; ".join(bad))
    return "\n".join(out_lines), tags


def merge(picks):
    """[(file, model, clip)] -> the same, one entry per file, strengths added up
    (as loading it twice would). Order of first use; both strengths 0 drop out."""
    merged = {}
    for f, model, clip in picks:
        m, c = merged.get(f, (0.0, 0.0))
        merged[f] = (m + model, c + clip)
    return [(f, m, c) for f, (m, c) in merged.items() if m or c]


# --- names -------------------------------------------------------------------------

def _norm(path):
    return path.replace("\\", "/").strip().strip("/").lower()


def _stem(path):
    p = _norm(path)
    for ext in EXTENSIONS:
        if p.endswith(ext):
            return p[:-len(ext)]
    return p


def _base(path):
    return _stem(path).rsplit("/", 1)[-1]


def matches(name, files):
    """Files (relative paths from the loras folder) that `name` points to.

    Case-insensitive, extension optional, `/` or `\\`. A full relative path
    wins; otherwise the file name alone, which may match in several folders.
    """
    key, stem = _norm(name), _stem(name)
    exact = [f for f in files if _norm(f) == key]
    if exact:
        return exact
    by_path = [f for f in files if _stem(f) == stem]
    if by_path:
        return by_path
    return [f for f in files if _base(f) == stem]


def resolve(name, files):
    """Exactly one file for `name`, or LookupError with a message a user can act on."""
    found = matches(name, files)
    if len(found) == 1:
        return found[0]
    if found:
        shown = ", ".join(sorted(insert_name(f, files) for f in found))
        raise LookupError(f'"{name}" fits several LoRAs: {shown}. Write the folder too.')
    near = difflib.get_close_matches(_stem(name), [_stem(f) for f in files], n=3, cutoff=0.6)
    hint = f" Did you mean {', '.join(near)}?" if near else ""
    raise LookupError(f'there\'s no LoRA "{name}" in the loras folder.{hint}')


def insert_name(path, files):
    """What autocomplete writes: the file name without extension, with folders only if needed."""
    base = _base(path)
    stem = path.replace("\\", "/")
    for ext in EXTENSIONS:
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break
    same = sum(1 for f in files if _base(f) == base)
    return stem.rsplit("/", 1)[-1] if same == 1 else stem


def describe(applied):
    """One line per applied LoRA for the node's `loras` output; clip None = no clip connected."""
    lines = []
    for name, model, clip in applied:
        if clip is None:
            lines.append(f"{name}: model {model:g}")
        elif model == clip:
            lines.append(f"{name}: {model:g}")
        else:
            lines.append(f"{name}: model {model:g}, clip {clip:g}")
    return "\n".join(lines)


def protect_hashes(text, marker):
    """'#' inside <lora:…> must survive AB Wildcard's comment cut (a LoRA file may have one)."""
    return TAG_RE.sub(lambda m: m.group(0).replace("#", marker), text)


def file_list_for_api(files):
    """Relative paths with '/' for the frontend."""
    return sorted((f.replace(os.sep, "/").replace("\\", "/") for f in files), key=str.lower)

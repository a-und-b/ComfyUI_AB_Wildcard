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


# --- LLM prep / apply ----------------------------------------------------------------
# Plan: docs/plan-llm-ops.md. Prep turns [gen …] markers into one request for a text
# generator (core Generate Text or any other); Apply splices its numbered-line answer
# back. Text outside the markers never passes through the LLM.

LLM_OPS = ["fill", "spin", "expand", "short", "fit", "do"]
LLM_DEFAULT_OPS = ["none", "spin", "expand", "short", "fit"]
LLM_SCOPES = ["full", "input", "none"]
# Directions: not parts to rewrite but guidance for every part; removed from the output.
LLM_DIRECTIONS = {"theme": "Theme", "angle": "Camera", "avoid": "Avoid"}
# [gen op (hint): content] — the hint sits before the first colon, so (word:1.2) in the content stays content
_MARKER = re.compile(r"\[gen(?:\s+(\w+))?\s*(?:\(([^():\]]*)\))?\s*:\s*([^\]]*?)\s*\]", re.IGNORECASE)

LLM_SYSTEM = (
    "You edit parts of a text-to-image prompt. Each task names one numbered part and "
    "what to do with it. Reply with exactly one line per task, in order, in the form "
    "\"N: new text\". Write only the new text for that part: no explanations, no quotes, "
    "no notes, no other lines."
)
LLM_NO_SLOTS_SYSTEM = "Reply with OK."

# Per-operation task. {c} = the part's original content, {n} = word limit for short,
# {ref} = what fit compares against, {h} = the marker's hint (do only; other ops append it).
LLM_TASKS = {
    "fill": "Write the concrete thing this asks for, in a few words (an actual example, not a "
            "description of the request): {c}",
    "spin": "Say \"{c}\" differently: same meaning and subject, but replace most of the words "
            "with different ones (synonyms, other details of the same thing).",
    "expand": "Rewrite \"{c}\" with more visual detail (materials, colors, light, setting): "
              "at least twice as many words, same subject. Keep its form: a list of short "
              "comma-separated phrases stays such a list, no full sentences.",
    "short": "Shorten \"{c}\" to at most {n} words. Keep the core, drop details.",
    "fit": "Check whether \"{c}\" belongs to the same era and world as {ref}. If it clearly does, "
           "write it unchanged. If anything about it is from another era or world (for example "
           "modern technology in a medieval scene), rewrite it to match: keep what kind of thing "
           "it is (a person stays a person, a place stays a place), change the details that clash.",
    "do": "Rewrite \"{c}\" as follows: {h}. Write it as a short image-prompt phrase.",
}
# What fit compares against, per scope; scope none gives the LLM nothing to fit to.
_FIT_REF = {"full": "the rest of the prompt", "input": "the context"}
# angle is also appended to the output by code (Apply): the camera belongs to the whole image,
# and a per-part rewrite has no reliable place for it.
_DIRECTION_NOTE = {"angle": " (added to the prompt as it is: don't repeat it, show each part as seen from there)"}


def _unknown_op(marker):
    return ValueError(f"AB LLM Prep: unknown operation {marker!r}. Use [gen: …], "
                      f"[gen spin|expand|short|fit|do: …] or [gen theme|angle|avoid: …].")


def llm_directions(text):
    """(text without direction markers, {name: [values]}).

    A removed marker takes one neighbouring comma with it, and its line if it stood alone.
    """
    found, out, pos = {}, [], 0
    for m in _MARKER.finditer(text or ""):
        name = (m.group(1) or "").lower()
        if name not in LLM_DIRECTIONS:
            continue
        if m.group(3):
            found.setdefault(name, []).append(m.group(3))
        start, end = m.start(), m.end()
        after = re.match(r"[ \t]*,[ \t]*", text[end:])
        before = re.search(r",[ \t]*$", text[pos:start])
        if after:
            end += after.end()
        elif before:
            start = pos + before.start()
        elif start == 0 or text[start - 1] in " \t":
            end += len(re.match(r"[ \t]*", text[end:]).group())  # "a [gen theme: x] b" -> "a b"
            if end == len(text) or text[end] == "\n":
                start = pos + len(text[pos:start].rstrip(" \t"))  # "a [gen theme: x]" -> "a"
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        line_end = len(text) if line_end < 0 else line_end
        if not text[line_start:start].strip() and not text[end:line_end].strip():
            start, end = max(line_start, pos), min(line_end + 1, len(text))  # whole line
            if line_end == len(text) and start > pos and text[start - 1] == "\n":
                start -= 1  # last line: take the newline before it instead
        out.append(text[pos:start])
        pos = end
    out.append((text or "")[pos:])
    return "".join(out), found


def llm_slots(text, default_op="none"):
    """Marker spans in text: [{op, content, hint, start, end}]. Direction markers are skipped.

    Text without markers is one slot with default_op ('none': no slots).
    """
    slots = []
    for m in _MARKER.finditer(text or ""):
        op = (m.group(1) or "fill").lower()
        if op in LLM_DIRECTIONS:
            continue
        if op not in LLM_OPS:
            raise _unknown_op(m.group(0))
        hint = (m.group(2) or "").strip()
        if op == "do" and not hint:
            raise ValueError(f"AB LLM Prep: {m.group(0)!r} needs an instruction: [gen do (what to do): …].")
        part = _protect(m.group(3))
        if not part["content"] and m.group(3).strip():
            raise ValueError(f"AB LLM Prep: {m.group(0)!r} has nothing left to rewrite once <…> tags and "
                             f"weights are set aside. Put the tag outside the marker.")
        slots.append({"op": op, "hint": hint, "start": m.start(), "end": m.end(), **part})
    if not slots and default_op != "none" and (text or "").strip():
        slots.append({"op": default_op, "hint": "", "start": 0, "end": len(text), **_protect(text.strip())})
    return slots


_TAG = re.compile(r"<[^<>]+>")  # <lora:name:0.8> and the like
_WEIGHT = re.compile(r"\(([^()]*?):\s*(-?\d+(?:\.\d+)?)\s*\)")  # (word:1.2)


def _protect(content):
    """What the LLM may see of a part, and what code keeps aside: {content, orig, weight, tags, dropped}.

    <…> tags come out and go back behind the answer. A weight on the whole part
    wraps the answer again. A weight on a piece of it can't follow a rewrite: it
    is dropped, and the report says so.
    """
    tags = _TAG.findall(content)
    rest = _TAG.sub("", content)
    rest = re.sub(r"\s*,\s*(?:,\s*)+", ", ", rest).strip(" \t,")  # "crown, <lora>" -> "crown"
    weight = ""
    whole = _WEIGHT.fullmatch(rest)
    if whole:
        rest, weight = whole.group(1).strip(), whole.group(2)
    dropped = [m.group(0) for m in _WEIGHT.finditer(rest)]
    rest = _WEIGHT.sub(lambda m: m.group(1).strip(), rest)
    return {"content": rest, "orig": content, "weight": weight, "tags": tags, "dropped": dropped}


def _restore(slot, new):
    """The answer with the part's whole weight and tags put back."""
    if slot.get("weight"):
        new = f"({new}:{slot['weight']})"
    return " ".join([new] + slot.get("tags", []))


def _task(i, slot, ref, directed):
    t = LLM_TASKS[slot["op"]].format(c=slot["content"], n=_word_limit(slot["content"]), ref=ref,
                                     h=slot["hint"].rstrip("."))
    extra = []
    if slot["hint"] and slot["op"] != "do":
        extra.append(f"Above all: {slot['hint'].rstrip('.')}.")
    if directed:
        extra.append("Follow the direction above.")
    if extra:
        t = t.rstrip() + ("" if t.rstrip()[-1:] in ".!?" else ".") + " " + " ".join(extra)
    return f"Task {i}: {t}"


def _word_limit(content):
    return max(1, len(content.split()) // 2)


def llm_prep(text, scope="full", default_op="none", context=""):
    """(system_prompt, prompt, plan) for one generator call."""
    for m in _MARKER.finditer(text or ""):  # fail on a typo before anything else
        if (m.group(1) or "fill").lower() not in LLM_OPS and m.group(1).lower() not in LLM_DIRECTIONS:
            raise _unknown_op(m.group(0))
    text, directions = llm_directions(text or "")
    slots = llm_slots(text, default_op)
    plan = {"text": text, "slots": slots, "append": ", ".join(directions.get("angle", []))}
    if not slots:
        if directions:
            plan["notes"] = ["no parts to edit: " + ", ".join(f"[gen {d}: …]" for d in directions)
                             + " removed, nothing else changed"]
        # ponytail: a trivial request instead of blocking; an ExecutionBlocker here would block the image too
        return LLM_NO_SLOTS_SYSTEM, "OK?", plan

    asked = scope
    whole = len(slots) == 1 and not text[:slots[0]["start"]].strip() and not text[slots[0]["end"]:].strip()
    if scope == "full" and whole:
        scope = "none"  # the slot is the whole prompt: showing it twice adds nothing
    blind = [i for i, s in enumerate(slots, 1)
             if s["op"] == "fit" and (scope == "none" or (scope == "input" and not (context or "").strip()))]
    if blind:  # shown in Apply's report: fit without anything to fit to is a guess
        plan["notes"] = [f"{i} fit: nothing to fit it to (scope {asked}"
                         + (", and the part is the whole prompt)" if whole and asked == "full" else ")")
                         for i in blind]

    if scope == "full":
        # the prompt as the model sees it: every marker shown as [N: content]
        parts, pos = [], 0
        for i, s in enumerate(slots, 1):
            parts += [text[pos:s["start"]], f"[{i}: {s['content']}]"]
            pos = s["end"]
        context = "".join(parts + [text[pos:]]).strip()
    elif scope == "none":
        context = ""
    context = (context or "").strip()

    # "Task N:", not "N:": a model that echoes the task list must not have it parsed as its answer
    ref = _FIT_REF.get(scope, "the rest of the prompt")
    tasks = [_task(i, s, ref, bool(directions)) for i, s in enumerate(slots, 1)]
    prompt = (f"The prompt (parts to edit marked [N: …]):\n{context}\n\n" if scope == "full" else
              f"Context:\n{context}\n\n" if context else "")
    if directions:
        prompt += "Direction for every task:\n" + "".join(
            f"- {LLM_DIRECTIONS[d]}: {'; '.join(v)}{_DIRECTION_NOTE.get(d, '')}\n"
            for d, v in ((d, directions[d]) for d in LLM_DIRECTIONS if d in directions)) + "\n"
    prompt += "Tasks:\n" + "\n".join(tasks)
    prompt += f"\n\nAnswer with exactly {len(slots)} line{'s' if len(slots) > 1 else ''}, " \
              + ", ".join(f"\"{i}: …\"" for i in range(1, len(slots) + 1)) + "."
    return LLM_SYSTEM, prompt, plan


_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)
_LINE = re.compile(r"^\W*?(\d+)\W*?[:.)]\s*(.*)$")  # tolerates "1:", "**1.**", "- 2)", "[3]:"


def parse_llm_lines(answer, count):
    """{slot number: answer} from a numbered-line answer; the first line per number wins."""
    found = {}
    for line in _THINK.sub("", answer or "").splitlines():
        m = _LINE.match(line.strip())
        if m and 1 <= int(m.group(1)) <= count and int(m.group(1)) not in found:
            found[int(m.group(1))] = m.group(2).strip().rstrip(",").strip().strip("\"'`*").strip()
    return found


def _append(text, extra):
    """text + ", " + extra at its end (camera from [gen angle: …])."""
    if not extra:
        return text
    base = text.rstrip(" \t\n,")
    return f"{base}{' ' if base.endswith('.') else ', '}{extra}" if base else extra


def llm_apply(plan, answer):
    """(text, report): each slot's answer spliced into the original text, or its fallback."""
    text, report = _apply_slots(plan, answer)
    if plan.get("append"):
        text = _append(text, plan["append"])
        report += f"\ncamera appended: \"{plan['append']}\""
    return text, report


def _apply_slots(plan, answer):
    text, slots = plan["text"], plan["slots"]
    if not slots:
        return text, "\n".join(plan.get("notes", []) + [
            "No [gen …] parts and default operation none: text passed through, answer ignored."])
    found = parse_llm_lines(answer, len(slots))
    out, pos, report = [], 0, list(plan.get("notes", []))
    for i, s in enumerate(slots, 1):
        new, why = found.get(i), None
        if new is None:
            why = "no line for it in the answer"
        elif not new:
            why = "empty answer"
        elif s["op"] == "short" and len(new) > len(s["content"]):
            why = "the short version is longer than the original"
        orig = s.get("orig", s["content"])
        if why:
            new = orig
            report.append(f"{i} {s['op']}: fallback ({why}), kept \"{new}\"")
        else:
            if not s["content"].endswith("."):
                new = new.rstrip(".")  # "cozy cabin at dusk." would end a comma list with a stray period
            new = _restore(s, new)
            report.append(f"{i} {s['op']}: \"{orig}\" → \"{new}\"")
            if s.get("dropped"):
                report.append(f"{i} {s['op']}: weight {', '.join(s['dropped'])} dropped, it can't follow a "
                              f"rewrite of part of the text. Weight the whole part: ([gen …]:1.2)")
        out += [text[pos:s["start"]], new]
        pos = s["end"]
    return "".join(out + [text[pos:]]), "\n".join(report)

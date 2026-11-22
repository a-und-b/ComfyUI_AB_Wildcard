#!/usr/bin/env python3
"""Run AB LLM Prep → Generate Text → AB LLM Apply on a live ComfyUI and report how the model did.

Each case is queued once per seed (the seed drives both the wildcard picks and the
sampler), the results land in a Markdown report plus the raw JSON next to it.
Standard library only: run it on any machine that reaches the ComfyUI server.

    python scripts/llm_eval.py                         # built-in cases, seeds 1-3, 9B encoder
    python scripts/llm_eval.py --seeds 1-5 --only theme,angle
    python scripts/llm_eval.py --clip qwen_3_4b.safetensors --clip-type lumina2
    python scripts/llm_eval.py --cases my_cases.json   # [{"name", "text", "scope"?, "default_operation"?, "context"?}]

Flags in the report are hints for reading, not verdicts:
  fallback   Apply kept the original (no usable answer)
  same       the answer equals the original
  bleed      the answer took words from the theme that the original didn't have
  weights    the original had (word:1.2) or <…> syntax, the result lost it (a weight on part of a
             part is dropped by design; the report says so, and the flag shows it)
  sentence   the answer reads like a sentence (is/are/was…) or adds many commas
  long       a short answer has more words than asked for
Plan: docs/plan-llm-ops.md
"""
import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid

CASES = [
    # baseline and the same prompt with directions: compare to see what theme/angle change
    {"name": "baseline",
     "text": "a portrait of a {knight|pilot}, [gen spin: red silk dress], [gen expand: castle], "
             "[gen fit: {astronaut in a space suit|neon street}], oil painting"},
    {"name": "theme",
     "text": "[gen theme: dark fairy tale, muted colors]\n"
             "a portrait of a {knight|pilot}, [gen spin: red silk dress], [gen expand: castle], "
             "[gen fit: {astronaut in a space suit|neon street}], oil painting"},
    {"name": "angle",
     "text": "[gen angle: low angle, 24mm wide lens]\n"
             "a portrait of a {knight|pilot}, [gen spin: red silk dress], [gen expand: castle], "
             "[gen short: a long winding road through misty hills at dawn], oil painting"},
    {"name": "avoid",
     "text": "[gen avoid: people, text]\na quiet harbour town, [gen expand: market square at noon], photo"},
    {"name": "hints",
     "text": "a portrait of a knight, [gen spin (more ornate): red silk dress], "
             "[gen expand (focus on light and weather): castle], [gen (one word): a profession fitting a knight]"},
    {"name": "do",
     "text": "a street scene, [gen do (a 1920s version, max 6 words): {astronaut|cowboy|robot}], film photo"},
    {"name": "fit-fits",
     "text": "a viking warrior on a longship, cold sea, [gen fit: horned iron helmet], realistic"},
    # weights and tags are kept by code: expect (…:1.3) and (…:1.2) around the answers, the lora behind
    # part 3, and a "dropped" note for (gold:1.2); a "weights" flag here means code lost something
    {"name": "weights",
     "text": "a queen, ([gen spin: red silk dress]:1.3), [gen spin: (velvet cape:1.2)], "
             "[gen expand: (gold:1.2) crown with rubies, <lora:detail:0.6>]"},
    {"name": "whole-expand", "default_operation": "expand",
     "text": "a cozy cabin, snowy forest, dusk, warm window light, film grain"},
]

def graph(case, seed, args):
    """API-format graph for one run. PreviewAny nodes 11-15 carry what the report needs."""
    gen = {
        "clip": ["3", 0], "system_prompt": ["2", 0], "prompt": ["2", 1], "max_length": args.max_length,
        "sampling_mode": "on", "sampling_mode.temperature": args.temperature, "sampling_mode.top_k": 64,
        "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05, "sampling_mode.repetition_penalty": 1.05,
        "sampling_mode.presence_penalty": 0.0, "sampling_mode.seed": seed,
        "thinking": False, "use_default_template": True, "mtp": "auto",
    }
    prep = {"text": ["1", 0], "scope": case.get("scope", "full"),
            "default_operation": case.get("default_operation", "none")}
    g = {
        "1": {"class_type": "ABWildcardNode", "inputs": {"text": case["text"], "seed": seed}},
        "2": {"class_type": "ABLLMPrep", "inputs": prep},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": args.clip, "type": args.clip_type,
                                                      "device": "default"}},
        "4": {"class_type": "TextGenerateV2", "inputs": gen},
        "5": {"class_type": "ABLLMApply", "inputs": {"plan": ["2", 2], "generated_text": ["4", 0]}},
        "11": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}},
        "12": {"class_type": "PreviewAny", "inputs": {"source": ["2", 1]}},
        "13": {"class_type": "PreviewAny", "inputs": {"source": ["4", 0]}},
        "14": {"class_type": "PreviewAny", "inputs": {"source": ["5", 0]}},
        "15": {"class_type": "PreviewAny", "inputs": {"source": ["5", 1]}},
    }
    if case.get("context"):
        g["6"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": case["context"]}}
        prep["context"] = ["6", 0]
    return g


OUT_NODES = {"picks": "11", "prompt": "12", "answer": "13", "text": "14", "report": "15"}


# --- reading the results -------------------------------------------------------------

_REPLACED = re.compile(r'^(\d+) (\w+): "(.*)" → "(.*)"$')
_FALLBACK = re.compile(r'^(\d+) (\w+): fallback \((.*)\), kept "(.*)"$')
_DIRECTION = re.compile(r"\[gen\s+(theme|angle|avoid)\s*:\s*([^\]]*?)\s*\]", re.IGNORECASE)
_STOP = {"with", "and", "the", "from", "into", "over", "under", "colors", "colour", "colours", "style"}


def words(s):
    return re.findall(r"[a-zA-Z][a-zA-Z'-]+", s.lower())


def parse_report(report):
    """[{n, op, old, new, fallback, why}] from AB LLM Apply's report; notes are skipped."""
    parts = []
    for line in (report or "").splitlines():
        m = _REPLACED.match(line)
        if m:
            parts.append({"n": int(m[1]), "op": m[2], "old": m[3], "new": m[4], "fallback": False, "why": ""})
            continue
        m = _FALLBACK.match(line)
        if m:
            parts.append({"n": int(m[1]), "op": m[2], "old": m[4], "new": m[4], "fallback": True, "why": m[3]})
    return parts


def flags(part, theme_words):
    old, new = part["old"], part["new"]
    if part["fallback"]:
        return ["fallback"]
    out = []
    if new.strip().lower() == old.strip().lower():
        out.append("same")
    if theme_words & (set(words(new)) - set(words(old))):
        out.append("bleed")
    tags = re.findall(r"<[^<>]+>", old)
    weights = re.findall(r":\s*(-?[\d.]+)\s*\)", old)
    if any(t not in new for t in tags) or any(not re.search(rf":\s*{re.escape(w)}\s*\)", new) for w in weights):
        out.append("weights")
    if new.count(",") > old.count(",") + 2 or re.search(r"\b(is|are|was|were)\b", new):
        out.append("sentence")
    if part["op"] == "short" and len(words(new)) > max(1, len(words(old)) // 2):
        out.append("long")
    return out


def theme_words(picks):
    return {w for m in _DIRECTION.finditer(picks or "") if m[1].lower() == "theme"
            for w in words(m[2]) if len(w) > 3 and w not in _STOP}


# --- talking to ComfyUI -----------------------------------------------------------------

def call(server, path, body=None, timeout=30):
    req = urllib.request.Request(server.rstrip("/") + path,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def queue(server, g, client_id):
    try:
        return call(server, "/prompt", {"prompt": g, "client_id": client_id})["prompt_id"], None
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            data = json.loads(body)
            errs = [f"{n}: {x['message']} {x.get('details', '')}".strip()
                    for n, ne in data.get("node_errors", {}).items() for x in ne.get("errors", [])]
            return None, "; ".join(errs) or data.get("error", {}).get("message", body)
        except (ValueError, AttributeError):
            return None, body[:300]


def wait(server, prompt_id, timeout):
    end = time.time() + timeout
    while time.time() < end:
        h = call(server, f"/history/{prompt_id}").get(prompt_id)
        if h:  # ComfyUI adds a prompt to the history once it finished, with or without error
            return h
        time.sleep(1)
    return None


def read_outputs(h):
    status = h.get("status", {})
    if status.get("status_str") == "error":
        for kind, msg in status.get("messages", []):
            if kind == "execution_error":
                return None, f"{msg.get('node_type')}: {msg.get('exception_message', '').strip()}"
        return None, "execution failed"
    outs = h.get("outputs", {})
    return {k: "".join(outs.get(n, {}).get("text", [""])) for k, n in OUT_NODES.items()}, None


# --- report ---------------------------------------------------------------------------------

def fmt_seeds(spec):
    seeds = []
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        seeds += range(int(a), int(b) + 1) if b else [int(a)]
    return seeds


def markdown(runs, args, started):
    lines = [f"# AB LLM eval — {started:%Y-%m-%d %H:%M}", "",
             f"Server `{args.server}` · encoder `{args.clip}` ({args.clip_type}) · temperature {args.temperature} "
             f"· seeds {args.seeds}", "",
             "| case | runs | parts | fallback | same | bleed | weights | sentence | long | errors |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    names = list(dict.fromkeys(r["case"] for r in runs))
    for name in names:
        rs = [r for r in runs if r["case"] == name]
        parts = [p for r in rs for p in r.get("parts", [])]
        count = lambda f: sum(f in p["flags"] for p in parts)  # noqa: E731
        lines.append(f"| {name} | {len(rs)} | {len(parts)} | " + " | ".join(
            str(count(f)) for f in ("fallback", "same", "bleed", "weights", "sentence", "long"))
            + f" | {sum(bool(r.get('error')) for r in rs)} |")
    for name in names:
        lines += ["", f"## {name}", ""]
        for r in (r for r in runs if r["case"] == name):
            lines.append(f"**seed {r['seed']}**" + (f" — error: {r['error']}" if r.get("error") else ""))
            if r.get("error"):
                lines.append("")
                continue
            lines += ["", "| # | op | original | new | flags |", "|---|---|---|---|---|"]
            for p in r["parts"]:
                cell = lambda s: s.replace("|", "\\|")  # noqa: E731
                lines.append(f"| {p['n']} | {p['op']} | {cell(p['old'])} | {cell(p['new'])} | {' '.join(p['flags'])} |")
            lines += ["", f"→ `{r['text']}`", ""]
            if args.show_prompt:
                lines += ["<details><summary>prompt and raw answer</summary>", "", "```text", r["prompt"], "",
                          "--- answer ---", r["answer"], "```", "", "</details>", ""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--clip", default="qwen3.5_9b_int8_convrot.safetensors")
    ap.add_argument("--clip-type", default="qwen_image")
    ap.add_argument("--seeds", default="1-3", help="e.g. 1-3 or 1,4,7")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--cases", help="JSON file with a list of cases instead of the built-in ones")
    ap.add_argument("--only", help="comma-separated case names")
    ap.add_argument("--out", help="Markdown report path (default: llm-eval-<time>.md)")
    ap.add_argument("--show-prompt", action="store_true", help="include Prep's prompt and the raw answer")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per run")
    args = ap.parse_args(argv)

    cases = json.load(open(args.cases, encoding="utf8")) if args.cases else CASES
    if args.only:
        keep = {n.strip() for n in args.only.split(",")}
        cases = [c for c in cases if c["name"] in keep]
    if not cases:
        sys.exit("no cases to run")
    try:
        call(args.server, "/object_info/ABLLMPrep")
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"can't reach ComfyUI at {args.server}: {e}")

    started, client_id = datetime.datetime.now(), str(uuid.uuid4())
    jobs = []
    for case in cases:  # queue everything first: the encoder stays loaded between runs
        for seed in fmt_seeds(args.seeds):
            pid, err = queue(args.server, graph(case, seed, args), client_id)
            jobs.append({"case": case["name"], "seed": seed, "id": pid, "error": err})
    print(f"queued {sum(bool(j['id']) for j in jobs)} of {len(jobs)} runs", file=sys.stderr)

    for j in jobs:
        if j["id"]:
            h = wait(args.server, j["id"], args.timeout)
            outs, err = read_outputs(h) if h else (None, f"no result after {args.timeout} s")
            j["error"] = err
            if outs:
                tw = theme_words(outs["picks"])
                j.update(outs, parts=[dict(p, flags=flags(p, tw)) for p in parse_report(outs["report"])])
        mark = "ERROR " + j["error"] if j.get("error") else " · ".join(
            f"{p['n']} {' '.join(p['flags']) or 'ok'}" for p in j.get("parts", []))
        print(f"{j['case']:>14} seed {j['seed']}: {mark}", file=sys.stderr)

    out = args.out or f"llm-eval-{started:%Y%m%d-%H%M}.md"
    with open(out, "w", encoding="utf8") as f:
        f.write(markdown(jobs, args, started))
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf8") as f:
        json.dump({"args": vars(args), "runs": jobs}, f, indent=2, ensure_ascii=False)
    print(f"report: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

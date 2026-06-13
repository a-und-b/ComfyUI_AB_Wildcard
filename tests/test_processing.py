"""Backend (text-processing) tests for the AB Wildcard node.

These pin down the behaviour referenced by the autocomplete truncation bug:
the engine resolves a *complete* wildcard token, so a truncated string
(e.g. produced by a stale widget value) is what fails — not the parser.
"""
import os

import nodes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(text, seed=1):
    return nodes.ABWildcardNode().process(text=text, seed=seed)[0]


def _subjects():
    path = os.path.join(ROOT, "wildcards", "examples", "subjects.txt")
    with open(path, encoding="utf8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def test_full_wildcard_resolves():
    """A complete __examples/subjects__ token resolves to a real entry."""
    out = run("__examples/subjects__", seed=7)
    assert "__" not in out, f"wildcard left unresolved: {out!r}"
    assert out in _subjects()


def test_truncated_token_stays_literal():
    """Regression guard for the Tab-completion bug: a truncated token
    (the stale, pre-completion widget value) does NOT resolve — reproducing
    the broken 'magenta, __ex' / '__examples/sub' output."""
    assert run("__examples/sub", seed=7) == "__examples/sub"


def test_reproducible_same_seed():
    """Same seed -> identical output (core ComfyUI expectation)."""
    text = "{red|blue|green}, __examples/subjects__"
    assert run(text, seed=123) == run(text, seed=123)


def test_choice_picks_a_listed_option():
    assert run("{red|blue|green}", seed=3) in {"red", "blue", "green"}


def test_bare_hash_truncates_line():
    """Documents finding #3: a bare '#' is treated as a comment and clips the
    rest of the line. (Hazard for the '#<seed>$$' tag syntax.)"""
    assert run("hello #world", seed=1) == "hello"


# --- Fix A: __Card Title__ resolves to a card prompt (incl. Prefix/Suffix) ---

def test_card_title_resolves_with_affixes():
    out = run("__Iron Knight__", seed=1)
    assert "__" not in out, f"card title left unresolved: {out!r}"
    assert "epic fantasy" in out          # from the card's Prefix
    assert "detailed armor textures" in out  # from the card's Suffix


def test_unknown_card_title_stays_literal():
    assert run("__Definitely Not A Card__", seed=1) == "__Definitely Not A Card__"


# --- Fix B: card Tags are exposed for <[...]> autocomplete ---

def test_tag_index_exposes_card_tags():
    loader = nodes.TagLoader(
        nodes.get_all_wildcard_paths(), {"ignore_paths": True, "verbose": False}
    )
    loader.build_index()
    tags = {t for e in loader.yaml_entries.values() for t in e.get("tags", [])}
    assert "warrior" in tags


def test_tag_aggregation_still_works():
    """<[warrior]> resolves to a tagged card's prompt."""
    out = run("<[warrior]>", seed=2)
    assert out and "<[" not in out and "__" not in out

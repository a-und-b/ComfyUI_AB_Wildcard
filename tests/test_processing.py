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


def test_lora_style_tags_stay_literal():
    """<lora:name:0.8> is not a wildcard: it must pass through untouched
    instead of being swallowed by the <scope:opts> tag-group path."""
    assert run("a cat <lora:myStyle:0.8>, sitting") == "a cat <lora:myStyle:0.8>, sitting"
    assert run("<embedding:bar> <hypernet:foo:0.5>") == "<embedding:bar> <hypernet:foo:0.5>"


def test_card_title_resolves_consistently_across_runs():
    """Regression: a card title must resolve to a prompt on EVERY run, not only
    the first. The GLOBAL_CACHE short-circuit in load_tags used to return the
    cached title before build_index() ran, so a fresh TagLoader had an empty
    yaml_entries and get_entry_details() fell back to the literal title."""
    nodes.GLOBAL_CACHE.clear()
    nodes.GLOBAL_INDEX.update({"built": False, "files": set(), "entries": {}})

    first = run("__Iron Knight__", seed=1)
    second = run("__Iron Knight__", seed=2)   # this run hits GLOBAL_CACHE

    for label, out in (("first", first), ("second", second)):
        assert "__" not in out, f"{label} run left the wildcard unresolved: {out!r}"
        assert out.strip().lower() != "iron knight", \
            f"{label} run fell back to the literal title: {out!r}"
        assert "epic fantasy" in out, f"{label} run lost the card Prefix: {out!r}"


# --- CSV wildcards (issues #1, #3) ---

import pytest


@pytest.fixture
def csv_wildcard(tmp_path, monkeypatch):
    """Point the engine at a temp dir holding one CSV wildcard."""
    def make(name, content):
        (tmp_path / name).write_text(content, encoding="utf8")
        monkeypatch.setattr(nodes, "get_all_wildcard_paths", lambda: [str(tmp_path)])
        nodes.GLOBAL_CACHE.clear()
        nodes.GLOBAL_INDEX.update({"built": False, "files": set(), "entries": {}})
    yield make
    nodes.GLOBAL_CACHE.clear()
    nodes.GLOBAL_INDEX.update({"built": False, "files": set(), "entries": {}})


def test_csv_row_sets_every_column_variable(csv_wildcard):
    """#1: one $var=value per line, so all columns land in variables."""
    csv_wildcard("test.csv", "color,shape,size\nred,circle,large\n")
    assert run("__test__ $color $shape $size", seed=1) == "red circle large"


def test_csv_values_may_contain_hash(csv_wildcard):
    """#3: '#' inside a CSV value (hex colour) must not cut later columns."""
    csv_wildcard("test.csv", "color,hex,shape\nred,deep red #FF0000,circle\n")
    assert run("__test__ $color $hex $shape", seed=1) == "red deep red #FF0000 circle"


def test_csv_token_mid_line_and_inline_comment_unchanged(csv_wildcard):
    """Token after other text still sets the first column; plain lines still lose '# comments'."""
    csv_wildcard("test.csv", "color,shape\nred,circle\n")
    assert run("a __test__ $color $shape b", seed=1) == "a red circle b"
    assert nodes.strip_inline_comment("hello # note") == "hello"
    assert nodes.strip_inline_comment("\n$hex=#FF0000\nplain # c\n") == "\n$hex=#FF0000\nplain\n"

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


# --- Tag matching is exact, not substring (issue #5) ---

CARDS = """\
Alice:
  Prompts:
    - "alice prompt"
  Tags:
    - female
    - image
Bob:
  Prompts:
    - "bob prompt"
  Tags:
    - male
    - mage
"""


def test_tag_match_is_not_substring(csv_wildcard):
    """#5: <[male]> must never pick the 'female' card, <[mage]> never 'image'."""
    csv_wildcard("cards.yaml", CARDS)
    for seed in range(1, 25):
        assert run("<[male]>", seed=seed) == "bob prompt"
        assert run("<[mage]>", seed=seed) == "bob prompt"
        assert run("<[male AND mage]>", seed=seed) == "bob prompt"
        assert run("<[female OR male]>", seed=seed) in {"alice prompt", "bob prompt"}


def test_tag_match_is_case_insensitive(csv_wildcard):
    csv_wildcard("cards.yaml", CARDS)
    assert run("<[MALE]>", seed=1) == "bob prompt"


# --- Variables in [if ...] conditions are checked, not text-substituted (issue #4) ---

def test_if_on_defined_variable_is_true():
    assert run("$a = x\n[if $a : yes|no]") == "yes"


def test_if_on_undefined_variable_is_false():
    assert run("[if $nope : yes|no]") == "no"


def test_if_variable_does_not_depend_on_its_value_appearing_in_text():
    """The value 'I exist' appears nowhere else in the prompt; the check must still pass."""
    assert run("$v = I exist\nfoo [if $v : found|missing]") == "foo found"


def test_if_equality_and_plain_usage_still_work():
    assert run("$a = x\n$a [if $a=x : y|n] [if $a=z : y|n]") == "x y n"
    assert run("$q = qq\n[if $q : the text $q was found]") == "the text qq was found"


# --- A variable can feed a tag query in a later assignment (issue #6) ---

CASCADE = """\
Color:
  Prompts: ["red", "blue"]
  Tags: [Color]
Red Shade:
  Prompts: ["crimson"]
  Tags: [Shade, Red]
Blue Shade:
  Prompts: ["navy"]
  Tags: [Shade, Blue]
"""


def test_variable_feeds_tag_query_in_later_assignment(csv_wildcard):
    csv_wildcard("colors.yaml", CASCADE)
    for seed in range(8):
        out = run("$c = <[color]>\n$s = <[ $c and Shade]>\n$s", seed=seed)
        assert out in ("crimson", "navy")
        # the shade must belong to the color that was drawn
        assert run("$c = <[color]>\n$s = <[ $c and Shade]>\n$c $s", seed=seed) in ("red crimson", "blue navy")


# --- Sequence features (1.2.0): stepping through options with the seed ---
# Set the seed widget to "increment" and every run takes the next step.

from itertools import product


def test_tilde_keeps_old_behaviour():
    """{~...} stays as it was: option number (seed mod count)."""
    assert [run("{~a|b|c}", seed=s) for s in range(6)] == ["a", "b", "c", "a", "b", "c"]


def test_combo_enumerates_every_combination_once():
    """{@...} blocks count like an odometer: the first one turns fastest,
    and n1*n2 consecutive seeds give every combination exactly once."""
    outs = [run("{@a|b} {@x|y|z}", seed=s) for s in range(6)]
    assert outs == ["a x", "b x", "a y", "b y", "a z", "b z"]
    # Plain {~} with sizes 2 and 4 would miss half the pairs; {@} doesn't.
    pairs = {run("{@a|b} {@1|2|3|4}", seed=s) for s in range(100, 108)}
    assert pairs == {f"{l} {n}" for l, n in product("ab", "1234")}


def test_combo_wraps_around():
    assert run("{@a|b} {@x|y|z}", seed=6) == run("{@a|b} {@x|y|z}", seed=0)


def test_deck_uses_each_option_once_per_round():
    """{!...} shuffles the options once per round of n runs: no repeats inside a round."""
    opts = ["a", "b", "c", "d", "e"]
    text = "{!" + "|".join(opts) + "}"
    for round_no in range(4):
        seeds = range(round_no * 5, round_no * 5 + 5)
        assert sorted(run(text, seed=s) for s in seeds) == opts
    # reproducible
    assert run(text, seed=7) == run(text, seed=7)
    # rounds aren't all the same order
    orders = {tuple(run(text, seed=r * 5 + i) for i in range(5)) for r in range(6)}
    assert len(orders) > 1


def test_sequential_multi_select_takes_consecutive_groups():
    """{~2$$...} takes options in order, two at a time."""
    outs = [run("{~2$$a|b|c|d}", seed=s) for s in range(3)]
    assert outs == ["a, b", "c, d", "a, b"]
    assert run("{~2$$ and $$a|b|c|d}", seed=1) == "c and d"


def test_sequence_modes_on_wildcard_files(csv_wildcard):
    csv_wildcard("col.txt", "red\ngreen\nblue\n")
    assert [run("__~col__", seed=s) for s in range(4)] == ["red", "green", "blue", "red"]
    assert [run("__~2$$col__", seed=s) for s in range(3)] == ["red, green", "blue, red", "green, blue"]
    assert sorted(run("__!col__", seed=s) for s in range(3, 6)) == ["blue", "green", "red"]
    combos = {run("__@col__ {@x|y}", seed=s) for s in range(6)}
    assert combos == {f"{c} {v}" for c, v in product(["red", "green", "blue"], "xy")}


def test_sequence_modes_do_not_disturb_other_random_picks():
    """Adding a {@}/{!} block must not change what plain {a|b|c} blocks pick."""
    plain = "{p|q|r|s} {1|2|3|4|5}"
    for s in range(10):
        with_seq = run("{@a|b} " + plain, seed=s).split(" ", 1)[1]
        with_deck = run("{!a|b|c} " + plain, seed=s).split(" ", 1)[1]
        assert with_seq == run(plain, seed=s) == with_deck


# --- $$ after @ and ! picks a group instead of silently falling back to random ---

def test_deck_multi_select_covers_every_option_per_round():
    """{!2$$a|b|c|d}: two runs of two options use all four options exactly once."""
    for rnd in range(4):
        got = []
        for s in (rnd * 2, rnd * 2 + 1):
            got += run("{!2$$a|b|c|d}", seed=s).split(", ")
        assert sorted(got) == ["a", "b", "c", "d"]
    assert run("{!2$$a|b|c|d}", seed=5) == run("{!2$$a|b|c|d}", seed=5)


def test_combo_multi_select_is_a_window_that_moves_with_the_counter():
    """{@2$$a|b|c}: window of two starting at the counter, first block turns fastest."""
    assert [run("{@2$$a|b|c}", seed=s) for s in range(4)] == ["a, b", "b, c", "c, a", "a, b"]
    assert run("{@2$$ and $$a|b|c}", seed=1) == "b and c"
    outs = {run("{@2$$a|b|c} {@x|y}", seed=s) for s in range(6)}
    assert len(outs) == 6


def test_sequence_file_multi_select_for_combo_and_deck(csv_wildcard):
    csv_wildcard("col.txt", "red\ngreen\nblue\n")
    assert [run("__@2$$col__", seed=s) for s in range(3)] == ["red, green", "green, blue", "blue, red"]
    got = []
    for s in (0, 1, 2):
        got += run("__!1$$col__", seed=s).split(", ")
    assert sorted(got) == ["blue", "green", "red"]

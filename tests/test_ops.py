"""Tests for ops.py, the logic behind the AB Utils nodes (tools.py)."""
import pytest

import nodes
import ops


# --- separators -------------------------------------------------------------------

def test_split_trims_and_drops_empty():
    assert ops.split(" b \r\n\na\n") == ["b", "a"]
    assert ops.split("a, b,,c", "comma") == ["a", "b", "c"]
    assert ops.split("a  b", "space") == ["a", "b"]
    assert ops.split("a;;b", "custom", ";;") == ["a", "b"]
    assert ops.split("a\nb", "custom", "") == ["a\nb"]


def test_join():
    assert ops.join(["a", "b"]) == "a\nb"
    assert ops.join(["a", "b"], "comma") == "a, b"
    assert ops.join(["a", "b"], "custom", " | ") == "a | b"


# --- operations ---------------------------------------------------------------------

def test_sort():
    assert ops.sort_lines(["b", "C", "a"]) == ["a", "b", "C"]
    assert ops.sort_lines(["b", "C", "a"], "Z → A") == ["C", "b", "a"]
    assert ops.sort_lines(["b", "C", "a"], "reverse order") == ["a", "C", "b"]


@pytest.mark.parametrize("how,text,expected", [
    ("contain", "RED", ["red apple", "Red car"]),
    ("don't contain", "apple", ["Red car"]),
    ("start with", "green", ["green apple"]),
    ("end with", "CAR", ["Red car"]),
    ("are exactly", "red car", ["Red car"]),
    ("match regex", "^red", ["red apple"]),
    ("contain", "", ["red apple", "green apple", "Red car"]),
])
def test_filter(how, text, expected):
    assert ops.filter_lines(["red apple", "green apple", "Red car"], how, text) == expected


def test_replace_plain_vs_regex():
    assert ops.replace_lines(["Red car", "red.dot"], "RED", "blue") == ["blue car", "blue.dot"]
    assert ops.replace_lines(["a.b"], ".", "\\1") == ["a\\1b"]  # plain: literal both ways
    assert ops.replace_lines(["cat 12"], r"(\w+) (\d+)", r"\2 \1", regex=True) == ["12 cat"]
    assert ops.replace_lines(["drop me", "keep"], "drop me", "") == ["keep"]
    assert ops.replace_lines(["a"], "", "x") == ["a"]


def test_bad_regex_message():
    with pytest.raises(ValueError, match="not a valid regular expression"):
        ops.filter_lines(["a"], "match regex", "(")


def test_remove_duplicates():
    assert ops.remove_duplicates(["a", "A", "b", "a"]) == ["a", "A", "b"]
    assert ops.remove_duplicates(["a", "A", "b"], ignore_case=True) == ["a", "b"]


def test_shuffle_stable_per_seed():
    items = [str(i) for i in range(20)]
    assert ops.shuffle_lines(items, 3) == ops.shuffle_lines(items, 3)
    assert ops.shuffle_lines(items, 3) != ops.shuffle_lines(items, 4)
    assert sorted(ops.shuffle_lines(items, 3), key=int) == items


# --- mix ------------------------------------------------------------------------------

def test_mix_chance_extremes():
    for seed in range(50):
        assert ops.mix(["a", "b", "", "d"], chances=[100, 0, 100, 100], seed=seed) == ["a", "d"]


def test_mix_chance_rate_and_unwired_slots():
    hits = sum(ops.mix(["a", "", "", ""], chances=[30, 100, 100, 100], seed=s) == ["a"] for s in range(2000))
    assert 500 < hits < 700
    for s in range(50):
        assert ("a" in ops.mix(["a", "", "", ""], chances=[50] * 4, seed=s)) == \
               ("a" in ops.mix(["a", "", "c", ""], chances=[50] * 4, seed=s))


def test_mix_weighted():
    picks = {tuple(ops.mix(["a", "", "c", "d"], weights=[1, 1, 0, 1], seed=s)) for s in range(100)}
    assert picks == {("a",), ("d",)}
    assert ops.mix(["", "", "", ""], weights=[1] * 4) == []


# --- wildcard files -------------------------------------------------------------------

@pytest.fixture
def base(tmp_path):
    yield str(tmp_path / "wildcards")
    ops.clear_wildcard_caches()


def test_save_modes_return_whole_file(base):
    _, path = ops.safe_path(base, "t/list.txt")
    assert ops.save_lines(path, "  a \n\nb\n", "replace file") == ["a", "b"]
    assert ops.save_lines(path, "b\nc\nc", "add new lines") == ["a", "b", "c"]
    assert ops.save_lines(path, "a", "add all lines") == ["a", "b", "c", "a"]
    assert ops.save_lines(path, "z", "replace file") == ["z"]
    assert ops.load_lines(path) == ["z"]


def test_save_appends_after_file_without_trailing_newline(base):
    _, path = ops.safe_path(base, "x")
    ops.save_lines(path, "a")
    with open(path, "w", encoding="utf8") as f:
        f.write("a\n# note\nb")
    assert ops.save_lines(path, "c", "add all lines") == ["a", "b", "c"]


@pytest.mark.parametrize("bad", ["", "  ", "../x", "a/../../x", "/etc/x", "~x", "@x", "!x", "a/../.."])
def test_safe_path_refuses_names_outside_base(base, bad):
    with pytest.raises(ValueError, match="can't be used as a file name"):
        ops.safe_path(base, bad)


def test_saved_file_resolves_in_wildcard_node(tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "base_path", str(tmp_path))
    nodes.GLOBAL_INDEX["built"] = True
    nodes.GLOBAL_CACHE["stale"] = ["x"]
    _, path = ops.safe_path(str(tmp_path / "wildcards"), "fresh")
    ops.save_lines(path, "only-entry")
    ops.clear_wildcard_caches()
    assert nodes.ABWildcardNode().process(text="__fresh__", seed=1)[0] == "only-entry"
    ops.clear_wildcard_caches()


# --- LLM prep / apply (docs/plan-llm-ops.md) ---------------------------------------

LLM_TEXT = "a knight, [gen spin: red silk dress], [gen expand: castle], [gen short: a very old and rusty iron sword], [gen: a mood]"


def test_llm_markers():
    slots = ops.llm_slots(LLM_TEXT)
    assert [(s["op"], s["content"]) for s in slots] == [
        ("spin", "red silk dress"), ("expand", "castle"),
        ("short", "a very old and rusty iron sword"), ("fill", "a mood")]
    assert ops.llm_slots("[GEN Fit:viking ]")[0]["op"] == "fit"
    with pytest.raises(ValueError, match="unknown operation"):
        ops.llm_slots("[gen foo: x]")


def test_llm_markers_survive_wildcard():
    """AB Wildcard resolves picks inside a marker and keeps the marker."""
    out = nodes.ABWildcardNode().process(text="[gen fit: {viking|viking}], [gen spin: dress]", seed=1)[0]
    assert out == "[gen fit: viking], [gen spin: dress]"


def test_llm_default_operation():
    assert ops.llm_slots("a cat") == []
    [s] = ops.llm_slots(" a cat ", "short")
    assert (s["op"], s["content"], s["start"], s["end"]) == ("short", "a cat", 0, 7)
    assert len(ops.llm_slots("[gen: x] y", "short")) == 1  # markers win over the default


def test_llm_prep_scopes():
    system, prompt, plan = ops.llm_prep(LLM_TEXT, "full")
    assert "N: new text" in system
    assert "a knight, [1: red silk dress], [2: castle]" in prompt
    assert "Shorten \"a very old and rusty iron sword\" to at most 3 words" in prompt
    assert prompt.endswith('"1: …", "2: …", "3: …", "4: …".')
    assert plan["text"] == LLM_TEXT and len(plan["slots"]) == 4
    assert "knight" not in ops.llm_prep(LLM_TEXT, "none")[1]
    _s, prompt, _p = ops.llm_prep(LLM_TEXT, "input", context="a dark forest")
    assert "Context:\na dark forest" in prompt and "knight" not in prompt


def test_llm_no_slots_still_sends_a_tiny_request():
    system, prompt, plan = ops.llm_prep("a cat, (sun:2)", "full", "none")
    assert (system, prompt) == ("Reply with OK.", "OK?")
    text, report = ops.llm_apply(plan, "OK")
    assert text == "a cat, (sun:2)" and "passed through" in report


def test_llm_parse_lines_tolerates_noise():
    answer = ("<think>\n1: wrong, still thinking\n</think>\nSure! Here you go:\n"
              "**1.** crimson satin gown\n- 2) \"a towering stone fortress\",\n3: rusty blade\n"
              "1: duplicate, ignored\n9: out of range\n```")
    assert ops.parse_llm_lines(answer, 3) == {
        1: "crimson satin gown", 2: "a towering stone fortress", 3: "rusty blade"}
    assert ops.parse_llm_lines("<think>never closed 1: x", 1) == {}


def test_llm_apply_splices_and_falls_back_per_slot():
    _s, _p, plan = ops.llm_prep(LLM_TEXT, "full")
    answer = "1: crimson satin gown\n3: a rusty sword that is much longer than the original was\n4: "
    text, report = ops.llm_apply(plan, answer)
    assert text == ("a knight, crimson satin gown, castle, a very old and rusty iron sword, a mood")
    lines = report.splitlines()
    assert lines[0] == '1 spin: "red silk dress" → "crimson satin gown"'
    assert "no line" in lines[1] and "longer than the original" in lines[2] and "empty" in lines[3]


def test_llm_apply_whole_text():
    _s, _p, plan = ops.llm_prep("a big red ball on a table", "full", "short")
    assert ops.llm_apply(plan, "1: red ball.")[0] == "red ball"

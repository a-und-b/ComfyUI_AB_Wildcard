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

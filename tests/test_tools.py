"""Tests for tools.py: wildcard Save/Load, AB Text Lines, AB Prompt Mix."""
import os

import folder_paths
import pytest

import nodes
import tools


@pytest.fixture
def root(tmp_path, monkeypatch):
    """ComfyUI base path in a temp dir; saved files land in <tmp>/wildcards."""
    monkeypatch.setattr(folder_paths, "base_path", str(tmp_path))
    yield tmp_path
    tools._clear_caches()


def save(text, name="t/list", mode="overwrite"):
    return tools.ABWildcardSave().save(text, name, mode)


def read(root, name="t/list"):
    return (root / "wildcards" / f"{name}.txt").read_text(encoding="utf8")


# --- Save -------------------------------------------------------------------

def test_save_overwrite_trims_and_drops_empty(root):
    assert save("  a \n\nb\n", name="t/list.txt") == ("  a \n\nb\n", "t/list")
    assert read(root) == "a\nb\n"
    save("c")
    assert read(root) == "c\n"


def test_save_append(root):
    save("a\nb")
    (root / "wildcards" / "t" / "list.txt").write_text("a\nb", encoding="utf8")  # no trailing newline
    save("b\nc", mode="append")
    assert read(root) == "a\nb\nb\nc\n"


def test_save_append_unique(root):
    save("a\nb")
    save("b\nc\nc\na\nd", mode="append unique")
    assert read(root) == "a\nb\nc\nd\n"


@pytest.mark.parametrize("bad", ["", "  ", "../x", "a/../../x", "/etc/x", "~x", "@x", "!x", "a/../.."])
def test_save_rejects_names_outside_folder(root, bad):
    with pytest.raises(ValueError, match="invalid name"):
        save("a", name=bad)
    assert not (root.parent / "x.txt").exists()


def test_save_clears_caches_and_wildcard_node_sees_file(root):
    nodes.GLOBAL_INDEX["built"] = True
    nodes.GLOBAL_CACHE["stale"] = ["x"]
    save("only-entry", name="fresh")
    assert nodes.GLOBAL_INDEX["built"] is False and not nodes.GLOBAL_CACHE
    assert nodes.ABWildcardNode().process(text="__fresh__", seed=1)[0] == "only-entry"


# --- Load -------------------------------------------------------------------

def test_load_reads_back_saved_file(root):
    _, key = save("a\n# comment\nb  # trailing\n", name="Dir/My")
    text, lines, count = tools.ABWildcardLoad().load(file="examples/colors", name=key)
    assert (text, lines, count) == ("a\nb", ["a", "b"], 2)


def test_load_bundled_file_from_combo(root):
    assert "examples/colors" in tools.ABWildcardLoad.INPUT_TYPES()["required"]["file"][0]
    _, lines, count = tools.ABWildcardLoad().load(file="examples/colors")
    assert count == len(lines) > 0


def test_load_unknown_file(root):
    with pytest.raises(ValueError, match="nope"):
        tools.ABWildcardLoad().load(file="nope")


# --- AB Text Lines ------------------------------------------------------------

def tl(text, **kw):
    return tools.text_lines(text, **kw)[1]


def test_lines_defaults_trim_drop_empty_dedupe():
    assert tl(" b \n\na\nb\n") == ["b", "a"]
    assert tl("b\nb", dedupe=False) == ["b", "b"]


def test_lines_split_and_join():
    joined, items = tools.text_lines("a, b,c", split_by=",", join_with=" | ")
    assert (joined, items) == ("a | b | c", ["a", "b", "c"])
    assert tl("a\tb", split_by="\\t") == ["a", "b"]
    assert tl("a\nb", split_by="") == ["a\nb"]


def test_lines_find_replace_plain_vs_regex():
    assert tl("Red car\nred.dot", find="RED", replace="blue") == ["blue car", "blue.dot"]
    assert tl("a.b", find=".", replace="\\1") == ["a\\1b"]  # plain: literal both ways
    assert tl("cat 12\ndog 3", find=r"(\w+) (\d+)", replace=r"\2 \1", regex=True) == ["12 cat", "3 dog"]
    assert tl("drop me\nkeep", find="drop me") == ["keep"]  # emptied items vanish


def test_lines_keep_and_drop():
    src = "red apple\ngreen apple\nRed car"
    assert tl(src, keep="red") == ["red apple", "Red car"]
    assert tl(src, drop="apple") == ["Red car"]
    assert tl(src, keep="^red", regex=True) == ["red apple"]


def test_lines_order():
    src = "b\nC\na"
    assert tl(src, order="sort") == ["a", "b", "C"]
    assert tl(src, order="sort desc") == ["C", "b", "a"]
    assert tl(src, order="reverse") == ["a", "C", "b"]
    many = "\n".join(str(i) for i in range(20))
    assert tl(many, order="shuffle", seed=3) == tl(many, order="shuffle", seed=3)
    assert tl(many, order="shuffle", seed=3) != tl(many, order="shuffle", seed=4)
    assert sorted(tl(many, order="shuffle", seed=3), key=int) == tl(many)


def test_lines_bad_regex_names_field():
    with pytest.raises(ValueError, match="'keep'"):
        tl("a", keep="(", regex=True)


def test_lines_node_outputs():
    assert tools.ABTextLines().run("a\nb\na") == ("a\nb", ["a", "b"], 2)


# --- AB Prompt Mix --------------------------------------------------------------

def mix(mode="each by chance", seed=0, separator=", ", **slots):
    return tools.ABPromptMix().mix(mode, seed, separator, **slots)[0]


def test_mix_chance_extremes():
    for seed in range(50):
        assert mix(seed=seed, text_1="a", weight_1=1.0, text_2="b", weight_2=0.0, text_4="d") == "a, d"


def test_mix_chance_roughly_matches_weight_and_ignores_unwired_slots():
    hits = sum(mix(seed=s, text_1="a", weight_1=0.3) == "a" for s in range(2000))
    assert 500 < hits < 700
    for s in range(50):  # wiring another slot does not change slot 1's outcome
        assert ("a" in mix(seed=s, text_1="a", weight_1=0.5)) == ("a" in mix(seed=s, text_1="a", weight_1=0.5, text_3="c"))


def test_mix_pick_one_weighted():
    picks = {mix("pick one weighted", seed=s, text_1="a", text_2="", weight_2=1.0, text_3="c", weight_3=0.0, text_4="d")
             for s in range(100)}
    assert picks == {"a", "d"}  # never the empty slot 2 or the zero-weight slot 3
    assert mix("pick one weighted", seed=7, text_1="a", text_2="b") == mix("pick one weighted", seed=7, text_1="a", text_2="b")
    assert mix("pick one weighted") == ""


def test_mix_separator_escape():
    assert mix(separator="\\n", text_1="a", text_2="b") == "a\nb"

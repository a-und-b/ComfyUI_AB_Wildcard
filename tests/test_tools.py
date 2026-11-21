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

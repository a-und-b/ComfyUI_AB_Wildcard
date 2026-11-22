"""Tests for loras.py, the logic behind AB LoRA Loader (tools.py)."""
import pytest

import loras
import nodes


def run(text, seed=1):
    return nodes.ABWildcardNode().process(text=text, seed=seed)[0]


# --- parsing ----------------------------------------------------------------------

@pytest.mark.parametrize("spec,expected", [
    ("x", ("x", 1.0, 1.0)),
    ("x:0.8", ("x", 0.8, 0.8)),
    ("x:0.8:0.5", ("x", 0.8, 0.5)),
    ("x:-1", ("x", -1.0, -1.0)),
    ("x:.5", ("x", 0.5, 0.5)),
    (" my style v2 : 1 ", ("my style v2", 1.0, 1.0)),
    ("a:b:0.7", ("a:b", 0.7, 0.7)),          # numbers from the right, the name keeps its colon
    ("x:1:2:3", ("x:1", 2.0, 3.0)),
    ("sub/dir/x.safetensors:0.8", ("sub/dir/x.safetensors", 0.8, 0.8)),
    ("0.8", ("0.8", 1.0, 1.0)),               # a lone number is a name
])
def test_parse_spec(spec, expected):
    assert loras.parse_spec(spec) == expected


def test_parse_spec_needs_a_name():
    with pytest.raises(ValueError):
        loras.parse_spec(" :0.8")


def test_extract_strips_tags_and_tidies_only_their_lines():
    text = "a cat, <lora:anime:0.8>, sitting\n<LORA:detail:1:0.5>\nkeep  ,this,\n<lyco:old>, dog"
    clean, tags = loras.extract(text)
    assert clean == "a cat, sitting\nkeep  ,this,\ndog"
    assert [(t.name, t.model, t.clip) for t in tags] == [
        ("anime", 0.8, 0.8), ("detail", 1.0, 0.5), ("old", 1.0, 1.0)]


@pytest.mark.parametrize("text,clean", [
    ("<lora:a:1> a cat", "a cat"),
    ("a cat <lora:a:1>", "a cat"),
    ("a, <lora:a:1>, <lora:b:1>, b", "a, b"),
    ("a <lora:a:1> b", "a b"),
    ("no tags, here ,", "no tags, here ,"),
    ("a < b > c", "a < b > c"),
    ("<lora:a>", ""),
])
def test_extract_clean_text(text, clean):
    assert loras.extract(text)[0] == clean


def test_extract_reports_a_tag_without_name():
    with pytest.raises(ValueError, match="<lora::0.8> has no name"):
        loras.extract("a <lora::0.8>")


def test_merge_adds_up_and_drops_zero():
    picks = [("a.safetensors", 0.5, 0.5), ("b.safetensors", 0.0, 0.0), ("a.safetensors", 0.3, 0.1)]
    assert loras.merge(picks) == [("a.safetensors", 0.8, 0.6)]


# --- names --------------------------------------------------------------------------

FILES = ["anime.safetensors", "sdxl/Detail.safetensors", "flux/detail.safetensors", "sub\\win.pt",
         "two.safetensors", "two.pt"]


@pytest.mark.parametrize("name,expected", [
    ("anime", "anime.safetensors"),
    ("ANIME.safetensors", "anime.safetensors"),
    ("flux/detail", "flux/detail.safetensors"),
    ("flux\\detail", "flux/detail.safetensors"),
    ("sub/win", "sub\\win.pt"),
    ("win", "sub\\win.pt"),
    ("two.pt", "two.pt"),
])
def test_resolve(name, expected):
    assert loras.resolve(name, FILES) == expected


def test_resolve_ambiguous_names_the_folders():
    with pytest.raises(LookupError, match=r'"detail" fits several LoRAs: flux/detail, sdxl/Detail\. Write the folder'):
        loras.resolve("detail", FILES)
    with pytest.raises(LookupError, match="fits several"):
        loras.resolve("two", FILES)


def test_resolve_missing_suggests():
    with pytest.raises(LookupError, match=r'no LoRA "animu".*Did you mean anime\?'):
        loras.resolve("animu", FILES)
    with pytest.raises(LookupError, match=r'no LoRA "zzz" in the loras folder\.$'):
        loras.resolve("zzz", FILES)


def test_insert_name():
    assert loras.insert_name("anime.safetensors", FILES) == "anime"
    assert loras.insert_name("sdxl/Detail.safetensors", FILES) == "sdxl/Detail"
    assert loras.insert_name("sub\\win.pt", FILES) == "win"


def test_describe():
    assert loras.describe([("anime", 0.8, 0.8), ("detail", 1.0, 0.5), ("x", 0.6, None)]) == \
        "anime: 0.8\ndetail: model 1, clip 0.5\nx: model 0.6"


# --- AB Wildcard keeps the tags intact ---------------------------------------------------

def test_wildcard_picks_lora_tags():
    assert run("{<lora:a:0.8>|<lora:a:0.8>} cat") == "<lora:a:0.8> cat"
    assert run("cyberpunk city [if cyberpunk : <lora:cyber:0.7>]") == "cyberpunk city <lora:cyber:0.7>"
    assert run("$s = <lora:oil:0.6>\na cat $s") == "a cat <lora:oil:0.6>"


def test_hash_inside_lora_tag_is_no_comment():
    assert run("a <lora:v#2:0.8>, cat # comment") == "a <lora:v#2:0.8>, cat"

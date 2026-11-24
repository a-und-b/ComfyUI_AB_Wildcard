"""AB Wildcard Extract: prompt -> wildcard files (docs/plan-wildcard-extract.md, Checks)."""
import pytest

import folder_paths
import nodes
import ops

PLAN_EXAMPLE = """$color = {red|blue|green}
a $color car in {Paris|Tokyo|Berlin}, {~morning|noon|night}

Moods:
- calm
- stormy"""


@pytest.fixture
def root(tmp_path, monkeypatch):
    """ComfyUI root with an empty wildcards/ folder; AB Wildcard reads from it."""
    monkeypatch.setattr(folder_paths, "base_path", str(tmp_path))
    ops.clear_wildcard_caches()
    yield tmp_path
    ops.clear_wildcard_caches()


def run(text, seed):
    return nodes.ABWildcardNode().process(text=text, seed=seed)[0]


def test_raw_output_is_the_input_and_feeds_extract(root):
    text, raw = nodes.ABWildcardNode().process(text=PLAN_EXAMPLE + "  // note", seed=1)
    assert raw == PLAN_EXAMPLE + "  // note"
    assert "{" not in text
    new, files, _ = ops.extract(raw)
    assert "__extracted/color__" in new and set(files) == {"color", "choice_1", "choice_2", "moods"}


# --- resolve check: decides whether nested blocks may move into files ----------------------

def test_file_line_with_block_resolves_like_inline(root):
    _, path = ops.safe_path(str(root / "wildcards"), "x/n")
    ops.save_lines(path, "a {b|c}\nd", "replace file")
    ops.clear_wildcard_caches()
    via_file = {run("__x/n__", s) for s in range(200)}
    inline = {run("{a {b|c}|d}", s) for s in range(200)}
    assert via_file == inline == {"a b", "a c", "d"}


# --- both forms, naming ------------------------------------------------------------------------

def test_plan_example():
    text, files, skipped = ops.extract(PLAN_EXAMPLE)
    assert text == ("$color = __extracted/color__\n"
                    "a $color car in __extracted/choice_1__, __~extracted/choice_2__\n\n"
                    "__extracted/moods__")
    assert files == {"moods": ["calm", "stormy"], "color": ["red", "blue", "green"],
                     "choice_1": ["Paris", "Tokyo", "Berlin"], "choice_2": ["morning", "noon", "night"]}
    assert skipped == []


@pytest.mark.parametrize("heading", ["Moods:", "# Moods", "## Moods", "**Moods**", "**Moods:**", "### Moods:"])
def test_heading_forms(heading):
    text, files, _ = ops.extract(f"{heading}\n- calm\n* stormy\n1. dark\n2) grey\nfoggy\n\nafter")
    assert text == "__extracted/moods__\n\nafter"
    assert files == {"moods": ["calm", "stormy", "dark", "grey", "foggy"]}


def test_list_ends_at_next_heading_and_empty_heading_stays():
    text, files, _ = ops.extract("Empty:\n\nA:\n- 1\nB:\n- 2")
    assert text == "Empty:\n\n__extracted/a__\n__extracted/b__"
    assert files == {"a": ["1"], "b": ["2"]}


def test_list_items_keep_inline_blocks():
    # lists go first; their items reach the file as they are (nested blocks resolve, see above)
    _, files, _ = ops.extract("Cars:\n- {red|blue} car\n- truck")
    assert files == {"cars": ["{red|blue} car", "truck"]}


def test_assignment_names_the_file_only_when_the_block_is_the_whole_value():
    text, files, _ = ops.extract("$Hair = {long|short}\n$mood = a {calm|wild} day")
    assert text == "$Hair = __extracted/hair__\n$mood = a __extracted/choice_1__ day"
    assert set(files) == {"hair", "choice_1"}


def test_positional_names_are_stable_across_runs(tmp_path):
    prompt = "{a|b} and {c|d}, {e|f}"
    assert ops.extract(prompt) == ops.extract(prompt)
    base = str(tmp_path / "wildcards")
    first = ops.save_extracted(base, "extracted", prompt)
    second = ops.save_extracted(base, "extracted", prompt)
    assert first == second
    assert first[1] == ["extracted/choice_1", "extracted/choice_2", "extracted/choice_3"]


# --- prefixes ----------------------------------------------------------------------------------

@pytest.mark.parametrize("block,ref", [
    ("{~a|b}", "__~extracted/choice_1__"),
    ("{!a|b}", "__!extracted/choice_1__"),
    ("{@a|b}", "__@extracted/choice_1__"),
    ("{2$$a|b|c}", "__2$$extracted/choice_1__"),
    ("{1-3$$a|b|c}", "__1-3$$extracted/choice_1__"),
    ("{2-$$a|b|c}", "__2-$$extracted/choice_1__"),
    ("{~2$$a|b|c}", "__~2$$extracted/choice_1__"),
    ("{!1-2$$a|b|c}", "__!1-2$$extracted/choice_1__"),
])
def test_prefixes_carry_over(block, ref):
    text, files, skipped = ops.extract(f"x {block} y")
    assert text == f"x {ref} y"
    assert files["choice_1"] == (["a", "b"] if block.endswith("a|b}") else ["a", "b", "c"])
    assert skipped == []


# --- nested blocks -----------------------------------------------------------------------------

def test_nested_block_stays_one_file_line():
    text, files, _ = ops.extract("{a {b|c}|d|e {f|{g|h}}}")
    assert text == "__extracted/choice_1__"
    assert files == {"choice_1": ["a {b|c}", "d", "e {f|{g|h}}"]}


def test_multiline_option_becomes_one_line():
    _, files, _ = ops.extract("{red car\n  in rain|blue}")
    assert files == {"choice_1": ["red car in rain", "blue"]}


# --- left inline, reported ---------------------------------------------------------------------

@pytest.mark.parametrize("block,why", [
    ("{5::a|b}", "weighted"),
    ("{50%a|b}", "chance"),
    ("{2$$ and $$a|b}", "custom separator"),
    ("{~2$$ and $$a|b}", "custom separator"),
    ("{a#b|c}", "#"),
    ("{alone}", "single option"),
    ("{}", "single option"),
    ("{a|}", "empty option"),
])
def test_skipped_kinds_stay_inline_and_are_reported(block, why):
    prompt = f"x {block} {{c|d}}"
    text, files, skipped = ops.extract(prompt)
    assert text == f"x {block} __extracted/choice_1__"  # positions count extracted blocks only
    assert list(files) == ["choice_1"]
    assert len(skipped) == 1 and skipped[0][0] == block and why in skipped[0][1]


def test_nested_weight_does_not_hold_back_the_outer_block():
    text, _, skipped = ops.extract("{a|{5::b|c}}")
    assert text == "__extracted/choice_1__" and skipped == []


def test_percent_that_is_no_chance_is_a_plain_choice():
    # AB Wildcard reads {N%…} as a chance only when the text before % is a number
    _, files, skipped = ops.extract("{half 50% off|silk}")
    assert files == {"choice_1": ["half 50% off", "silk"]} and skipped == []
    assert ops.extract("{100% cotton|silk}")[2] == [("{100% cotton|silk}", "chance")]


def test_unclosed_brace_is_left_alone():
    assert ops.extract("a {b|c") == ("a {b|c", {}, [])


# --- names -------------------------------------------------------------------------------------

@pytest.mark.parametrize("heading,name", [
    ("Hair Colors", "hair_colors"),
    ("  Mood / Light (dark)  ", "mood__light_dark"),
    ("Größe-2", "größe-2"),
    ("~@!", "list_1"),
])
def test_slug_rules(heading, name):
    _, files, _ = ops.extract(f"{heading}:\n- x")
    assert list(files) == [name]


def test_name_collisions_get_suffixes():
    prompt = "Color:\n- a\n\n# color\n- b\n\n$color = {c|d}\n\nchoice 1:\n- e\n\n{f|g}"
    text, files, _ = ops.extract(prompt)
    assert list(files) == ["color", "color_2", "choice_1", "color_3", "choice_1_2"]
    assert "$color = __extracted/color_3__" in text


# --- folder and modes ----------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "..", "../x", "a/../../x", "/etc", "/", "~x"])
def test_bad_folder_is_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="AB Wildcard Extract: .* can't be used as a file name"):
        ops.save_extracted(str(tmp_path / "wildcards"), bad, "{a|b}")


def test_folder_trailing_slash_and_subfolders(tmp_path):
    base = str(tmp_path / "wildcards")
    text, keys, _ = ops.save_extracted(base, "my/set/", "{a|b}")
    assert text == "__my/set/choice_1__" and keys == ["my/set/choice_1"]
    assert (tmp_path / "wildcards" / "my" / "set" / "choice_1.txt").read_text() == "a\nb\n"


def test_three_modes(tmp_path):
    base = str(tmp_path / "wildcards")
    path = tmp_path / "wildcards" / "extracted" / "choice_1.txt"
    ops.save_extracted(base, "extracted", "{a|b}")
    assert path.read_text() == "a\nb\n"
    ops.save_extracted(base, "extracted", "{b|c}", "add new lines")
    assert path.read_text() == "a\nb\nc\n"
    ops.save_extracted(base, "extracted", "{b|c}", "add all lines")
    assert path.read_text() == "a\nb\nc\nb\nc\n"
    ops.save_extracted(base, "extracted", "{x|y}")  # default: replace file
    assert path.read_text() == "x\ny\n"


# --- roundtrip through AB Wildcard ---------------------------------------------------------------

def _extract_into(root, prompt):
    text, _, _ = ops.save_extracted(str(root / "wildcards"), "extracted", prompt)
    ops.clear_wildcard_caches()
    return text


@pytest.mark.parametrize("prompt", [
    "$color = {red|blue}\na $color car",
    "in {Paris|Tokyo {by night|by day}}",
    "{!x|y|z}",
    "{2$$p|q|r}",
    "{1-2$$s|t}",
    "{~2$$i|j|k}",
])
def test_roundtrip_same_option_set(root, prompt):
    text = _extract_into(root, prompt)
    assert "{" not in text
    before = {run(prompt, s) for s in range(300)}
    after = {run(text, s) for s in range(300)}
    assert before == after


def test_roundtrip_sequences_identical_per_seed(root):
    prompt = "{~a|b|c} {@d|e} {@f|g|h} {~2$$i|j|k|l}"
    text = _extract_into(root, prompt)
    assert text.count("__") == 8
    for s in range(30):
        assert run(text, s) == run(prompt, s)

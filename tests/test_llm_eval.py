"""scripts/llm_eval.py reads AB LLM Apply's report: keep both in sync."""
import importlib.util
import os

import ops

_spec = importlib.util.spec_from_file_location(
    "llm_eval", os.path.join(os.path.dirname(__file__), "..", "scripts", "llm_eval.py"))
llm_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(llm_eval)


def test_report_round_trip_and_flags():
    text = "[gen theme: dark fairy tale]\na queen, [gen spin: (gold:1.2) crown], [gen short: a long road at dawn], [gen expand: hall]"
    _s, _p, plan = ops.llm_prep(text, "full")
    _t, report = ops.llm_apply(plan, "1: dark crown\n2: a long dusty road at dawn")
    parts = llm_eval.parse_report(report)
    assert [(p["n"], p["op"], p["fallback"]) for p in parts] == [(1, "spin", False), (2, "short", True), (3, "expand", True)]
    tw = llm_eval.theme_words(text)
    assert tw == {"dark", "fairy", "tale"}
    assert llm_eval.flags(parts[0], tw) == ["bleed", "weights"]
    assert llm_eval.flags(parts[1], tw) == ["fallback"]
    assert llm_eval.flags({"op": "short", "old": "a long road at dawn", "new": "a long road at dawn",
                           "fallback": False}, set()) == ["same", "long"]


def test_graph_wires_prep_generator_and_apply():
    args = type("A", (), {"max_length": 512, "temperature": 0.7, "clip": "c", "clip_type": "t"})
    g = llm_eval.graph({"name": "x", "text": "a", "scope": "input", "context": "ctx"}, 3, args)
    assert g["4"]["inputs"]["system_prompt"] == ["2", 0] and g["5"]["inputs"]["plan"] == ["2", 2]
    assert g["2"]["inputs"]["context"] == ["6", 0] and g["1"]["inputs"]["seed"] == 3
    assert llm_eval.fmt_seeds("1-3,7") == [1, 2, 3, 7]


def test_weights_flag_follows_the_weight_not_the_words():
    part = lambda old, new: {"op": "spin", "old": old, "new": new, "fallback": False}  # noqa: E731
    assert "weights" not in llm_eval.flags(part("(velvet cape:1.2)", "(plush mantle:1.2)"), set())
    assert "weights" not in llm_eval.flags(part("castle, <lora:d:0.6>", "big castle <lora:d:0.6>"), set())
    assert "weights" in llm_eval.flags(part("(gold:1.2) crown", "ornate crown"), set())
    assert "weights" in llm_eval.flags(part("castle, <lora:d:0.6>", "big castle"), set())


def test_sentence_flag_wants_lists_kept():
    part = lambda old, new: {"op": "expand", "old": old, "new": new, "fallback": False}  # noqa: E731
    assert "sentence" not in llm_eval.flags(part("castle", "stone fortress, gray walls, moonlit towers"), set())
    assert "sentence" in llm_eval.flags(part("cabin, forest, dusk", "a cabin in a forest at dusk"), set())
    assert "sentence" in llm_eval.flags(part("market", "a market where people trade"), set())

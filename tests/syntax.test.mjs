// Run: node tests/syntax.test.mjs   (plain assert, no framework)
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { tokenize, completionContext, completionText, variableNames, findMatches, replaceAll } from "../js/ab_syntax.js";

const toks = (t) => tokenize(t).tokens.map((k) => `${k.type}:${t.slice(k.start, k.end)}`);
const errs = (t) => tokenize(t).diagnostics.filter((d) => d.severity === "error").map((d) => `${d.message}@${d.start}`);
const diags = (t, known) => tokenize(t, known).diagnostics.map((d) => `${d.severity}:${d.message}`);

// --- valid constructs ---
assert.deepEqual(toks("$c = <[color]>"), ["varDef:$c", "assign:=", "tagDelim:<", "tagDelim:[", "tagDelim:]", "tagDelim:>"]);
assert.deepEqual(toks("a $c.upper b"), ["variable:$c.upper"]);
assert.deepEqual(toks("__a/b__ __2$$x__"), ["wildcard:__a/b__", "wildcard:__2$$x__"]);
assert.deepEqual(toks("__~f__ __!2$$f__"), ["wildcardSeq:__~f__", "wildcardSeq:__!2$$f__"]);
assert.deepEqual(toks("{a|b}"), ["brace:{", "pipe:|", "brace:}"]);
assert.deepEqual(toks("{!a|b}"), ["brace:{", "seqMode:!", "pipe:|", "brace:}"]);
assert.deepEqual(toks("{@2$$a|b|c}"), ["brace:{", "seqMode:@", "count:2$$", "pipe:|", "pipe:|", "brace:}"]);
assert.deepEqual(toks("{1-2$$ and $$a|b}"), ["brace:{", "count:1-2$$ and $$", "pipe:|", "brace:}"]);
assert.deepEqual(toks("{5::a|b}"), ["brace:{", "weight:5::", "pipe:|", "brace:}"]);
assert.deepEqual(toks("{30%a|b}"), ["brace:{", "prob:30%", "pipe:|", "brace:}"]);
assert.deepEqual(toks("{a|{b|c}}"), ["brace:{", "pipe:|", "brace:{", "pipe:|", "brace:}", "brace:}"]);
assert.deepEqual(toks("[if $a AND b: x | y]"), ["ifKw:[if", "variable:$a", "ifSep::", "pipe:|", "ifKw:]"]);
assert.deepEqual(toks("[clean: a, ,b]"), ["fn:[clean:", "fn:]"]);
assert.deepEqual(toks("[gen: a job] [GEN spin : red]"), ["fn:[gen:", "fn:]", "fn:[GEN spin :", "fn:]"]);
assert.deepEqual(toks("[gen fit: {a|b} $x]"), ["fn:[gen fit:", "brace:{", "pipe:|", "brace:}", "variable:$x", "fn:]"]);
assert.deepEqual(toks("[generic] [gen x]"), [], "no colon, not a marker");
assert.deepEqual(toks("[gen spin (more gothic): red] [gen theme: noir]"), ["fn:[gen spin (more gothic):", "fn:]", "fn:[gen theme:", "fn:]"]);
assert.deepEqual(toks("[gen short: (gold:1.2) crown]"), ["fn:[gen short:", "fn:]"], "a weight after the colon is content");
assert.deepEqual(toks("<[a AND (b OR --c)]>"),
  ["tagDelim:<", "tagDelim:[", "tagOp:AND", "tagOp:(", "tagOp:OR", "tagOp:--", "tagOp:)", "tagDelim:]", "tagDelim:>"]);
assert.deepEqual(toks("<lora:x:0.8> a < b > c"), ["angle:<lora:x:0.8>"]);
assert.deepEqual(toks("{2$$a|b}"), ["brace:{", "count:2$$", "pipe:|", "brace:}"]);

// --- comments (engine cuts at first // or #, __# and <# protected) ---
assert.deepEqual(toks("a # c {x"), ["comment:# c {x"]);
assert.deepEqual(toks("a // c"), ["comment:// c"]);
assert.deepEqual(toks("__#7$$n__ # x"), ["wildcard:__#7$$n__", "comment:# x"]);
assert.deepEqual(errs("a # {unclosed"), []);

// --- broken input ---
assert.deepEqual(errs("{!a|b"), ["Unclosed {@0"]);
assert.deepEqual(errs("a}"), ["Unmatched }@1"]);
assert.deepEqual(errs("[if $a x]"), ["[if needs a ':' after the condition@0"]);
assert.deepEqual(errs("[if $a: x"), ["Unclosed [if@0"]);
assert.deepEqual(errs("[clean: a"), ["Unclosed [clean: / [shuffle:@0"]);
assert.deepEqual(errs("x [gen spin: a"), ["Unclosed [gen@2"]);
assert.deepEqual(diags("[gen spinn: a] [gen expand: b]"), ['warning:Unknown [gen …] operation "spinn": use fill, spin, expand, short, fit, do, theme, angle, avoid']);
assert.deepEqual(diags("[gen do: a] [gen do (older): b]"), ["warning:[gen do …] needs an instruction: [gen do (what to do): …]"]);
assert.deepEqual(errs("<[a AND b"), ["Unclosed <[ … ]>@0"]);
assert.deepEqual(errs("[if $a: {x]"), ["Unclosed {@8"]);

// --- warnings ---
assert.deepEqual(diags("a $x b"), ["warning:$x is never assigned"]);
assert.deepEqual(diags("$x = 1\na $x"), []);
assert.deepEqual(diags("a $x", { globals: ["X"] }), []);
assert.deepEqual(diags("costs $5 $$"), []);
assert.deepEqual(diags("__a__ $x"), [], "wildcards may define variables");
assert.deepEqual(diags("{}"), ["warning:Empty {} has no options"]);
assert.deepEqual(diags("{3$$a|b}"), ["warning:Count 3 is larger than the 2 option(s)"]);
assert.deepEqual(diags("{2$$a|b}"), []);
assert.deepEqual(diags("{!a|b} {!a|b} {!a|c}"), ["warning:Identical {!…} blocks draw the same option"]);
assert.deepEqual(diags("{@a|{@b|c}}"), ["warning:{@…} inside another {@…} is not counted evenly"]);
assert.deepEqual(diags("see http://x.y"), ["info:'//' starts a comment, even in a URL"]);
assert.deepEqual(diags("{a // b\n|c}"), ["info:Inside a block this starts a comment; the rest of the line is ignored"]);
const known = { wildcards: ["Color", "examples/subjects"], tags: ["blue", "dark"] };
assert.deepEqual(diags("__color__ __~subjects__ __2$$examples/subjects__ __glob*__", known), []);
assert.deepEqual(diags("__nope__ __1-2$$nada__", known), ['warning:Unknown wildcard "nope"', 'warning:Unknown wildcard "nada"']);
assert.deepEqual(diags("<[blue AND dark]> <lora:x:0.8>", known), []);
assert.deepEqual(diags("<[blue AND $v OR purple]>", known), ['warning:Unknown tag "purple"']);

// --- invariants on every input: sorted, non-overlapping, in range ---
function checkShape(t) {
  let prev = 0;
  for (const k of tokenize(t).tokens) {
    assert.ok(k.start >= prev && k.end > k.start && k.end <= t.length, `bad token ${JSON.stringify(k)} in ${JSON.stringify(t)}`);
    prev = k.end;
  }
}

// --- autocomplete ---
const ctx = (t) => { const c = completionContext(t); return c && `${c.kind}:${c.query}`; };
assert.equal(ctx("$co"), "variable:co");
assert.equal(ctx("a $"), "variable:");
assert.equal(ctx("x {2$$"), null);
assert.equal(ctx("$5"), null);
assert.equal(ctx("<co"), "wildcard:co");
assert.equal(ctx("a < b"), null);
assert.equal(ctx("<lora:x"), null);
assert.equal(ctx("<[co"), "tag:co");
assert.equal(ctx("__co"), "wildcard:co");
assert.equal(ctx("$color.cl"), "modifier:cl");
assert.equal(ctx("costs $color."), null);
assert.equal(ctx("$color.upper.ti"), "modifier:ti");
assert.equal(ctx("plain text"), null);
{
  const t = "a $color.cl";
  const c = completionContext(t);
  assert.equal(t.slice(0, c.start) + completionText(c, "clean"), "a $color.clean");
  assert.equal(completionText(completionContext("x <co"), "colors"), "<colors>");
  assert.equal(completionText(completionContext("x __co"), "colors"), "__colors__");
  assert.equal(completionText(completionContext("x <[co"), "red"), "<[red]>");
  assert.equal(completionText(completionContext("x $c"), "color"), "$color");
}
assert.deepEqual(variableNames("a $late\n$late = {x}\n$Late = y", ["quality"]), ["quality", "late"]);
assert.deepEqual(variableNames("# $x = 1\n$y = 2"), ["y"]);

// --- find and replace ---
assert.deepEqual(findMatches("Red red RED", "red"), [{ start: 0, end: 3 }, { start: 4, end: 7 }, { start: 8, end: 11 }]);
assert.deepEqual(findMatches("abc", ""), []);
assert.deepEqual(findMatches("a {x|y} $c", "{x|"), [{ start: 2, end: 5 }]);
assert.deepEqual(findMatches("cost $5 or $$", "$$"), [{ start: 11, end: 13 }]);
assert.deepEqual(findMatches("aaa", "aa"), [{ start: 0, end: 2 }]);
assert.equal(replaceAll("$color $Color", "$color", "$tint"), "$tint $tint");
assert.equal(replaceAll("a b", "a", "$&$$"), "$&$$ b");
assert.equal(replaceAll("abc", "", "x"), "abc");

// --- every shipped example text: no diagnostics ---
const wfDir = new URL("../example_workflows/", import.meta.url);
let n = 0;
for (const f of readdirSync(wfDir).filter((f) => f.endsWith(".json"))) {
  const wf = JSON.parse(readFileSync(new URL(f, wfDir), "utf8"));
  for (const node of wf.nodes || []) {
    if (node.type !== "ABWildcardNode") continue;
    const text = node.widgets_values?.[0];
    if (typeof text !== "string") continue;
    checkShape(text);
    assert.deepEqual(diags(text), [], `${f}: ${JSON.stringify(text.slice(0, 60))}`);
    n++;
  }
}
assert.ok(n >= 6, `only ${n} example texts found`);

// --- README code blocks ---
const readme = readFileSync(new URL("../README.md", import.meta.url), "utf8");
for (const m of readme.matchAll(/`([^`\n]*[{$_<\[][^`\n]*)`/g)) checkShape(m[1]);

console.log(`ok (${n} example texts)`);

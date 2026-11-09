import json

from compass_pcp.harness.hb_analysis import analyze


def _write(path, rows):
    ex = []
    for pid, score, met in rows:
        ex.append({"prompt_id": pid, "score": score, "completion": [{"content": f"{pid}-{score}"}],
                   "rubric_items": [{"criteria_met": met, "points": 5, "tags": ["cluster:c1"]}]})
    path.write_text(json.dumps({"example_level": ex}))


def test_noninferiority_and_superiority(tmp_path):
    _write(tmp_path / "final.json", [(f"p{i}", 0.9, True) for i in range(40)])
    _write(tmp_path / "raw.json", [(f"p{i}", 0.6, i % 2 == 0) for i in range(40)])
    r = analyze(tmp_path, "final", "raw", ["raw"])["grader1"]
    assert r["HB1"]["noninferior"] and abs(r["HB1"]["diff"] - 0.5) < 1e-9
    assert abs(r["superiority"]["raw"]["diff"] - 0.3) < 1e-9 and r["superiority"]["raw"]["holm_significant"]
    assert r["by_cluster_vs_raw"]["c1"]["a_only"] == 20


def test_criterion_passes_only_when_every_repeat_passes(tmp_path):
    _write(tmp_path / "final.json", [("p0", 1.0, True), ("p0", 0.0, False)])
    _write(tmp_path / "raw.json", [("p0", 1.0, True)])
    r = analyze(tmp_path, "final", "raw", ["raw"])["grader1"]
    assert r["HB1"]["rate_a"] == 0.0 and r["scores"]["final"]["mean"] == 0.5

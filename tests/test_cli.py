import json

from twelve_angry_llms.cli import main


def test_export_command(tmp_path):
    annotated = tmp_path / "annotated.jsonl"
    row = {
        "id": "dp-0",
        "prompt": "p",
        "responses": ["r1", "r2", "r3"],
        "meta": {},
        "protocol": "scoring",
        "metric": "kendall",
        "ija": 0.9,
        "ija_pairwise": {"a|b": 0.9},
        "judge_values": {"a": [1, 2, 3], "b": [1, 3, 2]},
        "judge_models": {"a": "model-a", "b": "model-b"},
        "failures": [],
    }
    annotated.write_text(json.dumps(row) + "\n", encoding="utf-8")
    out = tmp_path / "pairs.jsonl"

    main(["export", "--input", str(annotated), "--output", str(out)])

    record = json.loads(out.read_text(encoding="utf-8"))
    # mean utilities [1, 2.5, 2.5] -> chosen is first max (index 1), rejected index 0
    assert record["chosen"] == "r2"
    assert record["rejected"] == "r1"
    assert record["prompt_ija"] == 0.9
    assert record["pair_agreement"] == 1.0

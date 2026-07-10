import json

import pytest

from twelve_angry_llms import (
    DatapointResult,
    JudgeAnnotation,
    PreferenceDatapoint,
    to_jsonl,
    to_records,
)


def annotation(judge: str, values) -> JudgeAnnotation:
    return JudgeAnnotation(
        judge=judge, model=f"model-{judge}", protocol="scoring", values=tuple(values), raw=""
    )


@pytest.fixture
def result(datapoint) -> DatapointResult:
    return DatapointResult(
        datapoint=datapoint,
        annotations=[
            annotation("a", [1, 2, 3, 5]),
            annotation("b", [2, 2, 3, 4]),
            annotation("c", [5, 1, 2, 3]),
        ],
        ija=0.42,
    )


class TestToRecords:
    def test_consensus_pair(self, result):
        record = to_records([result])[0]
        # mean utilities: [8/3, 5/3, 8/3, 4]; response 4 chosen, response 2 rejected
        assert record["chosen_index"] == 3
        assert record["rejected_index"] == 1
        assert record["chosen"] == result.datapoint.responses[3]
        assert record["rejected"] == result.datapoint.responses[1]
        assert record["prompt_ija"] == 0.42

    def test_pair_agreement(self, result):
        record = to_records([result])[0]
        # judges a and b prefer chosen(3) over rejected(1); c prefers it too
        assert record["pair_agreement"] == pytest.approx(1.0)

    def test_pair_agreement_ties_half(self, datapoint):
        result = DatapointResult(
            datapoint=datapoint,
            annotations=[annotation("a", [1, 2, 2, 3]), annotation("b", [3, 2, 2, 2])],
        )
        # consensus means: [2, 2, 2, 2.5] -> chosen=3, rejected=0 (first min)
        record = to_records([result])[0]
        assert record["chosen_index"] == 3
        # a prefers chosen (3>1) = 1.0; b ties (2 == ... wait: b values[3]=2, values[0]=3)
        # b prefers rejected -> 0.0; agreement = 0.5
        assert record["pair_agreement"] == pytest.approx(0.5)

    def test_meta_strategy(self, datapoint):
        dp = PreferenceDatapoint(
            prompt=datapoint.prompt,
            responses=datapoint.responses,
            meta={"chosen_index": 0, "rejected_index": 3},
        )
        result = DatapointResult(datapoint=dp, annotations=[annotation("a", [1, 2, 3, 4])])
        record = to_records([result], strategy="meta")[0]
        assert record["chosen_index"] == 0
        assert record["rejected_index"] == 3

    def test_meta_strategy_missing_indices(self, result):
        with pytest.raises(KeyError):
            to_records([result], strategy="meta")

    def test_unannotated_skipped(self, datapoint):
        empty = DatapointResult(datapoint=datapoint, annotations=[])
        assert to_records([empty]) == []
        with pytest.raises(ValueError):
            to_records([empty], skip_unannotated=False)

    def test_unknown_strategy(self, result):
        with pytest.raises(ValueError):
            to_records([result], strategy="argmax")


def test_to_jsonl_roundtrip(result, tmp_path):
    records = to_records([result])
    path = to_jsonl(records, tmp_path / "out" / "pairs.jsonl")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["prompt"] == result.datapoint.prompt

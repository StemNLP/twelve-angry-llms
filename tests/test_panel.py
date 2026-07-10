import pytest

from twelve_angry_llms import Judge, Panel, ScoringProtocol
from twelve_angry_llms.errors import JudgeError

from conftest import FakeClient, scores_json


def make_judge(name: str, script, **kwargs) -> Judge:
    kwargs.setdefault("api_retries", 0)
    return Judge(model=f"model-{name}", client=FakeClient(script), name=name, **kwargs)


class TestPanel:
    async def test_ija_hand_computed(self, datapoint):
        panel = Panel(
            [
                make_judge("a", scores_json(1, 2, 3, 4)),
                make_judge("b", scores_json(2, 3, 4, 5)),
                make_judge("c", scores_json(4, 3, 2, 1)),
            ]
        )
        result = await panel.annotate_one(datapoint, ScoringProtocol())
        # a-b agree perfectly (tau=1), each disagrees perfectly with c (tau=-1)
        assert result.ija == pytest.approx(-1 / 3)
        assert result.ija_pairwise == {
            "a|b": pytest.approx(1.0),
            "a|c": pytest.approx(-1.0),
            "b|c": pytest.approx(-1.0),
        }
        assert len(result.annotations) == 3
        assert not result.failures

    async def test_judge_failure_recorded(self, datapoint):
        def explode(messages):
            raise RuntimeError("provider down")

        panel = Panel(
            [
                make_judge("a", scores_json(1, 2, 3, 4)),
                make_judge("b", scores_json(1, 2, 3, 4)),
                make_judge("broken", explode),
            ]
        )
        result = await panel.annotate_one(datapoint, ScoringProtocol())
        assert [f.judge for f in result.failures] == ["broken"]
        assert "provider down" in result.failures[0].error
        # IJA still computed from the two surviving judges
        assert result.ija == pytest.approx(1.0)

    async def test_parse_repair_second_attempt(self, datapoint):
        judge = make_judge("a", ["not json at all", scores_json(1, 2, 3, 4)])
        annotation = await judge.annotate(datapoint, ScoringProtocol())
        assert annotation.values == (1.0, 2.0, 3.0, 4.0)
        assert judge.client.calls == 2

    async def test_parse_failure_exhausts_repairs(self, datapoint):
        judge = make_judge("a", "never json", parse_repairs=1)
        with pytest.raises(JudgeError):
            await judge.annotate(datapoint, ScoringProtocol())
        assert judge.client.calls == 2  # initial + one repair

    async def test_annotate_many_preserves_order(self, datapoint):
        panel = Panel(
            [
                make_judge("a", scores_json(1, 2, 3, 4)),
                make_judge("b", scores_json(1, 2, 3, 4)),
            ]
        )
        results = await panel.annotate([datapoint] * 5, ScoringProtocol(), concurrency=2)
        assert len(results) == 5
        assert all(r.datapoint.id == datapoint.id for r in results)

    async def test_diagnostics(self, datapoint):
        panel = Panel(
            [
                make_judge("a", scores_json(1, 2, 3, 4)),
                make_judge("b", scores_json(1, 2, 3, 4)),
            ]
        )
        results = await panel.annotate([datapoint] * 3, ScoringProtocol())
        diag = panel.diagnostics(results)
        assert diag.krippendorff_alpha == pytest.approx(1.0)
        assert diag.judge_correlation == {"a|b": pytest.approx(1.0)}
        assert diag.n_datapoints == 3
        assert diag.n_annotations == 6
        assert diag.n_failures == 0

    def test_needs_two_judges(self):
        with pytest.raises(ValueError):
            Panel([make_judge("solo", "x")])

    def test_unique_names(self):
        with pytest.raises(ValueError):
            Panel([make_judge("a", "x"), make_judge("a", "y")])

    def test_sync_wrapper(self, datapoint):
        panel = Panel(
            [
                make_judge("a", scores_json(1, 2, 3, 4)),
                make_judge("b", scores_json(4, 3, 2, 1)),
            ]
        )
        results = panel.annotate_sync([datapoint], ScoringProtocol())
        assert results[0].ija == pytest.approx(-1.0)

import pytest

from twelve_angry_llms.errors import ParseError
from twelve_angry_llms.protocols import RankingProtocol, ScoringProtocol


class TestScoringProtocol:
    protocol = ScoringProtocol()

    def test_messages_contain_everything(self, datapoint):
        messages = self.protocol.build_messages(datapoint)
        assert messages[0]["role"] == "system"
        user = messages[1]["content"]
        assert datapoint.prompt in user
        for i, response in enumerate(datapoint.responses, start=1):
            assert f"[Response {i}]" in user
            assert response in user

    def test_parse_clean_json(self):
        assert self.protocol.parse('{"scores": [1, 3, 5, 2]}', 4) == (1.0, 3.0, 5.0, 2.0)

    def test_parse_fenced_json(self):
        raw = 'Here you go:\n```json\n{"scores": [2, 4]}\n```\nHope that helps.'
        assert self.protocol.parse(raw, 2) == (2.0, 4.0)

    def test_parse_prose_wrapped(self):
        raw = 'After careful review I conclude {"scores": [5, 1]} as stated.'
        assert self.protocol.parse(raw, 2) == (5.0, 1.0)

    def test_wrong_length(self):
        with pytest.raises(ParseError):
            self.protocol.parse('{"scores": [1, 2, 3]}', 4)

    def test_out_of_scale(self):
        with pytest.raises(ParseError):
            self.protocol.parse('{"scores": [0, 3]}', 2)
        with pytest.raises(ParseError):
            self.protocol.parse('{"scores": [3, 9]}', 2)

    def test_non_numeric(self):
        with pytest.raises(ParseError):
            self.protocol.parse('{"scores": ["good", 3]}', 2)

    def test_no_json(self):
        with pytest.raises(ParseError):
            self.protocol.parse("Response 1 is best, then 3, then 2, then 4.", 4)

    def test_repair_messages_include_failed_reply(self, datapoint):
        messages = self.protocol.repair_messages(datapoint, "garbled")
        assert messages[-2] == {"role": "assistant", "content": "garbled"}
        assert "could not be parsed" in messages[-1]["content"]


class TestRankingProtocol:
    protocol = RankingProtocol()

    def test_messages_contain_everything(self, datapoint):
        user = self.protocol.build_messages(datapoint)[1]["content"]
        assert datapoint.prompt in user
        assert "best to worst" in user

    def test_parse_converts_to_utilities(self):
        # ranking [2, 1, 3]: response 2 is best (utility 3), then 1, then 3
        assert self.protocol.parse('{"ranking": [2, 1, 3]}', 3) == (2.0, 3.0, 1.0)

    def test_identity_ranking(self):
        assert self.protocol.parse('{"ranking": [1, 2, 3, 4]}', 4) == (4.0, 3.0, 2.0, 1.0)

    def test_not_a_permutation(self):
        with pytest.raises(ParseError):
            self.protocol.parse('{"ranking": [1, 1, 3]}', 3)
        with pytest.raises(ParseError):
            self.protocol.parse('{"ranking": [0, 1, 2]}', 3)

    def test_wrong_length(self):
        with pytest.raises(ParseError):
            self.protocol.parse('{"ranking": [1, 2]}', 3)

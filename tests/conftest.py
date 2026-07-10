import json

import pytest

from twelve_angry_llms import PreferenceDatapoint


class FakeClient:
    """Scriptable stand-in for a provider client.

    ``script`` may be a fixed string, a list of strings consumed in call
    order, or a callable taking the messages and returning a string.
    """

    def __init__(self, script):
        self.script = script
        self.calls = 0

    async def complete(self, *, model, messages, temperature=0.0, max_tokens=1024):
        self.calls += 1
        if callable(self.script):
            return self.script(messages)
        if isinstance(self.script, list):
            return self.script.pop(0)
        return self.script


def scores_json(*scores) -> str:
    return json.dumps({"scores": list(scores)})


def ranking_json(*order) -> str:
    return json.dumps({"ranking": list(order)})


@pytest.fixture
def datapoint() -> PreferenceDatapoint:
    return PreferenceDatapoint(
        prompt="What causes tides?",
        responses=(
            "The gravitational pull of the moon and sun.",
            "Mostly wind patterns over the ocean.",
            "The moon's gravity, with a smaller solar contribution.",
            "Tides are caused by the rotation of the earth alone.",
        ),
        id="dp-0",
    )

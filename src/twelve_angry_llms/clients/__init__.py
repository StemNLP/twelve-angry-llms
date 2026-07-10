from .base import LLMClient, Message, Usage
from .openai_compat import OpenAICompatibleClient

__all__ = [
    "LLMClient",
    "Message",
    "Usage",
    "OpenAICompatibleClient",
    "AnthropicClient",
]


def __getattr__(name: str):
    # AnthropicClient is behind the optional 'anthropic' extra; import lazily
    # so `from twelve_angry_llms.clients import OpenAICompatibleClient` works
    # without it installed.
    if name == "AnthropicClient":
        from .anthropic_client import AnthropicClient

        return AnthropicClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

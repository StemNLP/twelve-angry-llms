import pytest

from twelve_angry_llms.clients.openai_compat import OpenAICompatibleClient, OpenRouterClient


class TestOpenAICompatibleClient:
    def test_requires_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAICompatibleClient()

    def test_custom_key_env(self, monkeypatch):
        monkeypatch.setenv("MY_PROVIDER_KEY", "sk-test")
        client = OpenAICompatibleClient(api_key_env="MY_PROVIDER_KEY")
        assert client._client.api_key == "sk-test"

    def test_cache_salt_reflects_routing(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        plain = OpenAICompatibleClient()
        pinned = OpenAICompatibleClient(
            extra_body={"provider": {"order": ["together"], "allow_fallbacks": False}}
        )
        assert plain.cache_salt != pinned.cache_salt
        assert "together" in pinned.cache_salt


class TestOpenRouterClient:
    def test_preconfigured(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        client = OpenRouterClient()
        assert "openrouter.ai" in str(client._client.base_url)
        assert client._client.api_key == "sk-or-test"

    def test_key_never_read_from_openai_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-wrong")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterClient()

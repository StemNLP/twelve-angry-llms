from twelve_angry_llms.cache import CachedClient, ResponseCache

from conftest import FakeClient


class TestResponseCache:
    def test_roundtrip(self, tmp_path):
        cache = ResponseCache(tmp_path / "cache.sqlite")
        key = ResponseCache.key("m", [{"role": "user", "content": "hi"}], 0.0, 100)
        assert cache.get(key) is None
        cache.put(key, "hello")
        assert cache.get(key) == "hello"

    def test_key_sensitivity(self):
        messages = [{"role": "user", "content": "hi"}]
        base = ResponseCache.key("m", messages, 0.0, 100)
        assert ResponseCache.key("other-model", messages, 0.0, 100) != base
        assert ResponseCache.key("m", messages, 0.7, 100) != base
        assert ResponseCache.key("m", [{"role": "user", "content": "yo"}], 0.0, 100) != base
        assert ResponseCache.key("m", messages, 0.0, 100) == base

    def test_salt_changes_key(self):
        messages = [{"role": "user", "content": "hi"}]
        base = ResponseCache.key("m", messages, 0.0, 100)
        pinned = ResponseCache.key("m", messages, 0.0, 100, salt='{"provider": "together"}')
        assert pinned != base

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "cache.sqlite"
        key = ResponseCache.key("m", [], 0.0, 1)
        ResponseCache(path).put(key, "kept")
        assert ResponseCache(path).get(key) == "kept"


class TestCachedClient:
    async def test_second_call_served_from_disk(self, tmp_path):
        inner = FakeClient("the answer")
        client = CachedClient(inner, tmp_path / "cache.sqlite")
        messages = [{"role": "user", "content": "question"}]

        first = await client.complete(model="m", messages=messages)
        second = await client.complete(model="m", messages=messages)

        assert first == second == "the answer"
        assert inner.calls == 1
        assert (client.hits, client.misses) == (1, 1)

    async def test_different_requests_miss(self, tmp_path):
        inner = FakeClient("x")
        client = CachedClient(inner, tmp_path / "cache.sqlite")
        await client.complete(model="m", messages=[{"role": "user", "content": "a"}])
        await client.complete(model="m", messages=[{"role": "user", "content": "b"}])
        assert inner.calls == 2

    async def test_client_cache_salt_separates_entries(self, tmp_path):
        # same model + messages, but clients pinned to different providers
        # (different cache_salt) must not share cache entries
        path = tmp_path / "cache.sqlite"
        messages = [{"role": "user", "content": "q"}]

        inner_a, inner_b = FakeClient("from together"), FakeClient("from deepinfra")
        inner_a.cache_salt = '{"provider": "together"}'
        inner_b.cache_salt = '{"provider": "deepinfra"}'

        a = await CachedClient(inner_a, path).complete(model="m", messages=messages)
        b = await CachedClient(inner_b, path).complete(model="m", messages=messages)
        assert (a, b) == ("from together", "from deepinfra")
        assert inner_a.calls == inner_b.calls == 1

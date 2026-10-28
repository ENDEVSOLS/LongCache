"""Tests for the OpenAI direct integration (CachedOpenAI)."""

from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from longcache.config import CacheConfig


from tests.helpers import HashEmbedder, trusted_context
from tests.test_integrations_v2 import completion


def _make_mock_embedder():
    return HashEmbedder()


@patch("longcache.core.cache.LocalEmbedder")
class TestCachedOpenAI:
    def _cached_client(self, MockEmbedder, mock_openai_client):
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        from longcache.integrations.openai import CachedOpenAI

        client = CachedOpenAI(
            mock_openai_client,
            config=CacheConfig(backend="memory", similarity_threshold=0.9),
            context=trusted_context(),
        )
        client.cache.embedder = mock_embedder
        return client

    def test_create_caches_result(self, MockEmbedder):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = completion("fake completion")

        client = self._cached_client(MockEmbedder, mock_client)

        messages = [{"role": "user", "content": "What is RAG?"}]

        # CachedOpenAI.chat is a CachedChatCompletions wrapper — its own
        # .create() proxies through to the underlying client's
        # chat.completions.create(), so the cached call site is client.chat.create().
        result1 = client.chat.create(model="gpt-4o", messages=messages)
        assert result1 == completion("fake completion")
        assert mock_client.chat.completions.create.call_count == 1

        result3 = client.chat.completions.create(model="gpt-4o", messages=messages)
        assert result3 == result1
        assert mock_client.chat.completions.create.call_count == 1

        result2 = client.chat.create(model="gpt-4o", messages=messages)
        assert result2 == result1
        assert mock_client.chat.completions.create.call_count == 1

    def test_model_and_system_prompt_isolate_entries(self, MockEmbedder):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = completion("result")
        client = self._cached_client(MockEmbedder, mock_client)
        first = [{"role": "system", "content": "brief"}, {"role": "user", "content": "question"}]
        second = [{"role": "system", "content": "long"}, {"role": "user", "content": "question"}]
        client.chat.completions.create(model="a", messages=first)
        client.chat.completions.create(model="b", messages=first)
        client.chat.completions.create(model="a", messages=second)
        client.chat.completions.create(model="a", messages=first)
        assert mock_client.chat.completions.create.call_count == 3

    def test_multi_turn_and_unhandled_options_bypass_cache(self, MockEmbedder):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = completion("result")
        client = self._cached_client(MockEmbedder, mock_client)
        multi_turn = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        for _ in range(2):
            client.chat.completions.create(model="a", messages=multi_turn)
            client.chat.completions.create(
                model="a", messages=[{"role": "user", "content": "question"}], top_p=0.5
            )
        assert mock_client.chat.completions.create.call_count == 4

    def test_error_response_is_not_cached(self, MockEmbedder):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = {"error": "temporary"}
        client = self._cached_client(MockEmbedder, mock_client)
        messages = [{"role": "user", "content": "question"}]
        client.chat.completions.create(model="a", messages=messages)
        client.chat.completions.create(model="a", messages=messages)
        assert mock_client.chat.completions.create.call_count == 2

    def test_sdk_response_type_survives_sqlite_restart(self, MockEmbedder, tmp_path):
        pytest.importorskip("openai")
        from openai.types.chat import ChatCompletion

        result = ChatCompletion.model_validate(
            {
                "id": "chat-1",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"content": "answer", "role": "assistant"},
                    }
                ],
                "created": 1,
                "model": "test-model",
                "object": "chat.completion",
            }
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = result
        MockEmbedder.return_value = _make_mock_embedder()
        from longcache.integrations.openai import CachedOpenAI

        config = CacheConfig(backend="sqlite", db_path=str(tmp_path / "openai.db"), ttl_seconds=0)
        client = CachedOpenAI(
            mock_client, config=config, context=trusted_context(), response_codec="openai-sdk-v1"
        )
        messages = [{"role": "user", "content": "question"}]
        first = client.chat.completions.create(model="test-model", messages=messages)
        client.cache.close()
        restarted = CachedOpenAI(
            mock_client, config=config, context=trusted_context(), response_codec="openai-sdk-v1"
        )
        second = restarted.chat.completions.create(model="test-model", messages=messages)
        restarted.cache.close()
        assert isinstance(first, ChatCompletion)
        assert isinstance(second, ChatCompletion)
        assert second.choices[0].message.content == "answer"
        assert mock_client.chat.completions.create.call_count == 1

    def test_extract_query_from_messages(self, MockEmbedder):
        client = self._cached_client(MockEmbedder, MagicMock())
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is the refund policy?"},
        ]
        assert client._extract_query(messages) == "What is the refund policy?"

    def test_extract_query_fallback_when_no_user_message(self, MockEmbedder):
        client = self._cached_client(MockEmbedder, MagicMock())
        messages = [{"role": "system", "content": "You are helpful."}]
        # Falls back to stringifying all messages
        result = client._extract_query(messages)
        assert "system" in result

    def test_proxy_attributes(self, MockEmbedder):
        mock_client = MagicMock()
        mock_client.some_attribute = "proxied_value"
        client = self._cached_client(MockEmbedder, mock_client)
        assert client.some_attribute == "proxied_value"

    async def test_acreate_caches_result(self, MockEmbedder):
        mock_client = MagicMock()
        mock_client.chat.completions.acreate = AsyncMock(
            return_value=completion("fake async completion")
        )

        client = self._cached_client(MockEmbedder, mock_client)
        messages = [{"role": "user", "content": "What is async RAG?"}]

        result1 = await client.chat.acreate(model="gpt-4o", messages=messages)
        assert result1 == completion("fake async completion")
        assert mock_client.chat.completions.acreate.call_count == 1

        result2 = await client.chat.acreate(model="gpt-4o", messages=messages)
        assert result2 == completion("fake async completion")
        assert mock_client.chat.completions.acreate.call_count == 1

    async def test_async_client_uses_awaited_create(self, MockEmbedder):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=completion("async"))
        client = self._cached_client(MockEmbedder, mock_client)
        messages = [{"role": "user", "content": "question"}]
        first = await client.chat.completions.create(model="model", messages=messages)
        second = await client.chat.completions.create(model="model", messages=messages)
        assert first == second
        assert mock_client.chat.completions.create.await_count == 1

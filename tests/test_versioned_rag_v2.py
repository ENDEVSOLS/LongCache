"""RAG integration acceptance, including a document update during generation."""

import asyncio
import pytest
from examples.versioned_rag import DocumentSnapshot, answer
from tests.test_policy_acceptance_v2 import cache  # noqa: F401


@pytest.mark.asyncio
async def test_changed_document_never_reuses_previous_answer(cache):  # noqa: F811
    question = "What is the return window?"
    old = DocumentSnapshot("revision-1", "Returns accepted within 30 days.")
    new = DocumentSnapshot("revision-2", "Returns accepted within 14 days.")
    assert await answer(cache, question, old) == old.text
    assert await answer(cache, question, old) == old.text
    assert await answer(cache, question, new) == new.text
    assert await answer(cache, question, new) == new.text
    assert cache.stats(context=old.context("shop", "alice"))["hits"] == 1
    assert cache.stats(context=new.context("shop", "alice"))["hits"] == 1
    assert cache.invalidate_by_tag(old.document_id, context=old.context("shop", "alice")) == 1
    assert cache.get(question, context=new.context("shop", "alice")) == new.text
    assert cache.get(question, context=new.context("other-shop", "alice")) is None


@pytest.mark.asyncio
async def test_document_change_during_generation_preserves_snapshot(cache):  # noqa: F811
    question = "What is the return window?"
    old = DocumentSnapshot("revision-1", "Returns accepted within 30 days.")
    new = DocumentSnapshot("revision-2", "Returns accepted within 14 days.")
    started, release = asyncio.Event(), asyncio.Event()

    async def generate(snapshot):
        started.set()
        await release.wait()
        return snapshot.text

    pending = asyncio.create_task(answer(cache, question, old, generate=generate))
    await started.wait()
    assert await answer(cache, question, new) == new.text
    release.set()
    assert await pending == old.text
    assert await answer(cache, question, new) == new.text
    assert await answer(cache, question, old) == old.text

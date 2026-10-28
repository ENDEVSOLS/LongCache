"""Executable changing-document RAG workflow with an extractive fixture generator.

The retrieved snapshot supplies both the answer and its immutable cache identity.
Replace the fixture generator with your provider without changing that boundary.
"""

from dataclasses import dataclass
from longcache import LongCache, RequestContext, RequestScope


@dataclass(frozen=True)
class DocumentSnapshot:
    revision: str
    text: str
    document_id: str = "returns-policy"

    def context(self, tenant: str, user: str) -> RequestContext:
        return RequestContext(
            scope=RequestScope(tenant, user),
            operation="rag.answer",
            provider="extractive-fixture",
            provider_scope="local",
            model="extractive-v1",
            prompt_version="returns-v1",
            corpus_version=self.revision,
            document_versions={self.document_id: self.revision},
        )


async def answer(
    cache: LongCache,
    question: str,
    snapshot: DocumentSnapshot,
    *,
    tenant="shop",
    user="alice",
    generate=None,
):
    """Use the same retrieved snapshot for lookup, generation and admission."""
    context = snapshot.context(tenant, user)
    decision = await cache.alookup(question, context=context)
    if decision.outcome == "HIT":
        return decision.response
    value = snapshot.text if generate is None else await generate(snapshot)
    await cache.aset(question, value, context=context, tags=[snapshot.document_id])
    return value

"""Public-interface fixtures with stable vectors and explicit identities."""

from hashlib import sha256
import numpy as np
from longcache import BaseEmbedder, EmbeddingDescriptor, RequestContext, RequestScope


class HashEmbedder(BaseEmbedder):
    dimension = 384
    descriptor = EmbeddingDescriptor("test-hash", "sha256-v1", 384)

    def embed(self, text):
        seed = int.from_bytes(sha256(text.encode()).digest()[:8], "big")
        vector = np.random.default_rng(seed).normal(size=384).astype(np.float32)
        return vector / np.linalg.norm(vector)


def trusted_context(**kwargs):
    return RequestContext(
        **(
            dict(
                scope=RequestScope("test-tenant", "test-user"),
                operation="test.answer",
                provider="fixture",
                provider_scope="fixture-v1",
                model="fixture-model",
                prompt_version="v1",
            )
            | kwargs
        )
    )

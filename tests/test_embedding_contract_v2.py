import numpy as np
import pytest
from longcache.core.base_embedder import BaseEmbedder
from longcache.core.embedding_contract import EmbeddingDescriptor, EmbeddingGuard


class TinyEmbedder(BaseEmbedder):
    revision = "fixture-v1"

    @property
    def dimension(self):
        return 2

    @property
    def descriptor(self):
        return EmbeddingDescriptor("fixture", self.revision, 2)

    def embed(self, text):
        return np.array([1, 0], dtype=np.float32)


def test_valid_single_empty_and_batch():
    guard = EmbeddingGuard(TinyEmbedder())
    assert guard.embed("q").shape == (2,)
    assert guard.batch([]).shape == (0, 2)
    assert guard.batch(["a", "b"]).shape == (2, 2)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([1, 0], dtype=np.float64),
        np.array([0, 0], dtype=np.float32),
        np.array([np.nan, 0], dtype=np.float32),
        np.array([1, 0, 0], dtype=np.float32),
    ],
)
def test_bad_vectors_rejected(bad):
    embedder = TinyEmbedder()
    embedder.embed = lambda _: bad
    with pytest.raises(ValueError):
        EmbeddingGuard(embedder).embed("q")


def test_revision_drift_rejected_at_same_dimension():
    embedder = TinyEmbedder()
    guard = EmbeddingGuard(embedder)
    guard.embed("q")
    embedder.revision = "fixture-v2"
    with pytest.raises(ValueError, match="generation changed"):
        guard.embed("q")


@pytest.mark.parametrize(
    "bad",
    [
        np.empty((0, 2), dtype=np.float64),
        np.array([[np.inf, 0]], dtype=np.float32),
        np.array([[1, 0], [0, 1]], dtype=np.float32)[:, ::-1],
    ],
)
def test_invalid_batch_contract_rejected(bad):
    embedder = TinyEmbedder()
    embedder.embed_batch = lambda texts: bad
    with pytest.raises(ValueError):
        EmbeddingGuard(embedder).batch([] if len(bad) == 0 else ["q"] * len(bad))

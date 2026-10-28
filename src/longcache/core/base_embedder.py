"""Abstract base class for all embedder implementations."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):
    """
    Abstract interface for embedding models.

    All embedders must produce L2-normalized float32 vectors
    of consistent dimension. The dimension is used to initialize
    the FAISS index and must not change between calls.
    """

    thread_safe = False

    @property
    def descriptor(self):
        """Declare a stable model/revision contract for persisted vector compatibility."""
        raise ValueError("Custom embedders must supply an EmbeddingDescriptor")

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Returns:
            L2-normalized float32 numpy array of shape (dimension,).
        """
        ...

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed multiple texts. Default: loop over embed().
        Override for batched efficiency.

        Returns:
            float32 array of shape (len(texts), dimension).
        """
        import numpy as np

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.stack([self.embed(t) for t in texts])

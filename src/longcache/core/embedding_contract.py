"""Validated embedding generations; equal dimensions do not imply compatible models."""

from dataclasses import asdict, dataclass
import threading
from weakref import WeakKeyDictionary
import numpy as np
from .identity import digest


@dataclass(frozen=True)
class EmbeddingDescriptor:
    model: str
    revision: str
    dimension: int
    preprocessing: str = "text-v1"
    normalization: str = "l2-float32-v1"

    def __post_init__(self):
        if type(self.dimension) is not int or self.dimension < 1:
            raise ValueError("Embedding dimension must be a positive integer")
        if not all(
            type(v) is str and v.strip() for v in (self.model, self.revision, self.preprocessing)
        ):
            raise ValueError("Embedding model, revision and preprocessing must be explicit")
        if self.normalization != "l2-float32-v1":
            raise ValueError("Unsupported normalization contract")

    @property
    def fingerprint(self):
        return digest(asdict(self))


def validate_vector(vector, dimension):
    if not isinstance(vector, np.ndarray) or vector.dtype != np.float32:
        raise ValueError("Embeddings must be float32 numpy arrays")
    if vector.shape != (dimension,) or not vector.flags.c_contiguous:
        raise ValueError("Invalid embedding shape or contiguity")
    if not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector)) - 1.0) > 1e-3:
        raise ValueError("Embeddings must be finite and L2-normalized")
    return vector.copy()


_locks = WeakKeyDictionary()
_registry_lock = threading.Lock()


class EmbeddingGuard:
    def __init__(self, embedder):
        self.embedder = embedder
        with _registry_lock:
            self.lock = _locks.setdefault(embedder, threading.RLock())
        self._descriptor = None

    @property
    def descriptor(self):
        with self.lock:
            current = self.embedder.descriptor
            if (
                not isinstance(current, EmbeddingDescriptor)
                or current.dimension != self.embedder.dimension
            ):
                raise ValueError("Invalid embedder descriptor")
            if self._descriptor is not None and self._descriptor != current:
                raise ValueError("Embedding generation changed during cache lifetime")
            self._descriptor = current
            return current

    def embed(self, text):
        with self.lock:
            descriptor = self.descriptor
            return validate_vector(self.embedder.embed(text), descriptor.dimension)

    def batch(self, texts):
        with self.lock:
            dimension = self.descriptor.dimension
            vectors = self.embedder.embed_batch(texts)
            if (
                not isinstance(vectors, np.ndarray)
                or vectors.shape != (len(texts), dimension)
                or vectors.dtype != np.float32
                or not vectors.flags.c_contiguous
            ):
                raise ValueError("Invalid embedding batch shape")
            for vector in vectors:
                validate_vector(vector, dimension)
            return vectors.copy()

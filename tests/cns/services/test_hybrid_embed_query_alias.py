"""Unit tests for HybridEmbeddingsProvider.embed_query compatibility alias."""

import numpy as np
import pytest

from clients.hybrid_embeddings_provider import HybridEmbeddingsProvider


def test_embed_query_returns_1d_list_from_numpy_vector():
    provider = HybridEmbeddingsProvider.__new__(HybridEmbeddingsProvider)
    provider.encode_realtime = lambda text: np.array([0.1, 0.2, 0.3], dtype=np.float16)

    result = provider.embed_query("hello")

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


def test_embed_query_flattens_2d_numpy_output():
    provider = HybridEmbeddingsProvider.__new__(HybridEmbeddingsProvider)
    provider.encode_realtime = lambda text: np.array([[0.1, 0.2]], dtype=np.float16)

    result = provider.embed_query("hello")

    assert isinstance(result, list)
    assert result == pytest.approx([0.1, 0.2], rel=1e-3, abs=1e-3)

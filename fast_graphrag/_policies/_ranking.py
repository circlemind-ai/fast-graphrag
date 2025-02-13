from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix

from ._base import BaseRankingPolicy


class RankingPolicy_WithThreshold(BaseRankingPolicy):  # noqa: N801
    @dataclass
    class Config:
        threshold: float = field(default=0.05)
        max_entities: int = field(default=128)

    config: Config = field()

    def __call__(self, scores: csr_matrix) -> csr_matrix:
        # Remove scores below threshold
        scores.data[scores.data < self.config.threshold] = 0
        if scores.nnz >= self.config.max_entities:
            smallest_indices = np.argpartition(scores.data, -self.config.max_entities)[: -self.config.max_entities]
            scores.data[smallest_indices] = 0
        scores.eliminate_zeros()

        return scores


class RankingPolicy_TopK(BaseRankingPolicy):  # noqa: N801
    @dataclass
    class Config:
        top_k: int = field(default=10)

    top_k: Config = field()

    def __call__(self, scores: csr_matrix) -> csr_matrix:
        assert scores.shape[0] == 1, "TopK policy only supports batch size of 1"
        if scores.nnz <= self.config.top_k:
            return scores

        smallest_indices = np.argpartition(scores.data, -self.config.top_k)[: -self.config.top_k]
        scores.data[smallest_indices] = 0
        scores.eliminate_zeros()

        return scores


class RankingPolicy_Elbow(BaseRankingPolicy):  # noqa: N801
    def __call__(self, scores: csr_matrix) -> csr_matrix:
        assert scores.shape[0] == 1, "Elbow policy only supports batch size of 1"
        if scores.nnz <= 1:
            return scores

        sorted_scores = np.sort(scores.data)

        # Compute elbow
        diff = np.diff(sorted_scores)
        elbow = np.argmax(diff) + 1

        smallest_indices = np.argpartition(scores.data, elbow)[:elbow]
        scores.data[smallest_indices] = 0
        scores.eliminate_zeros()

        return scores


class RankingPolicy_WithConfidence(BaseRankingPolicy):  # noqa: N801
    def __call__(self, scores: csr_matrix) -> csr_matrix:
        raise NotImplementedError("Confidence policy is not supported yet.")

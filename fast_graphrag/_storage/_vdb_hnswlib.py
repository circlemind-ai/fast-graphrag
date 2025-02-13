import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import hnswlib
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._types import GTEmbedding, GTId, TScore
from fast_graphrag._utils import logger

from ._base import BaseVectorStorage


@dataclass
class HNSWVectorStorageConfig:
    ef_construction: int = field(default=128)
    M: int = field(default=64)
    ef_search: int = field(default=96)
    num_threads: int = field(default=-1)


@dataclass
class HNSWVectorStorage(BaseVectorStorage[GTId, GTEmbedding]):
    RESOURCE_NAME = "hnsw_index_{}.bin"
    RESOURCE_METADATA_NAME = "hnsw_metadata.pkl"
    INITIAL_MAX_ELEMENTS = 128000
    config: HNSWVectorStorageConfig = field()  # type: ignore
    _index: Any = field(init=False, default=None)  # type: ignore
    _metadata: Dict[GTId, Dict[str, Any]] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return self._index.get_current_count()

    @property
    def max_size(self) -> int:
        return self._index.get_max_elements() or self.INITIAL_MAX_ELEMENTS

    async def upsert(
        self,
        ids: Iterable[GTId],
        embeddings: Iterable[GTEmbedding],
        metadata: Union[Iterable[Dict[str, Any]], None] = None,
    ) -> None:
        ids = list(ids)
        embeddings = np.array(list(embeddings), dtype=np.float32)
        metadata = list(metadata) if metadata else None

        assert (len(ids) == len(embeddings)) and (metadata is None or (len(metadata) == len(ids))), (
            "ids, embeddings, and metadata (if provided) must have the same length"
        )

        if self.size + len(embeddings) >= self.max_size:
            new_size = self.max_size * 2
            while self.size + len(embeddings) >= new_size:
                new_size *= 2
            self._index.resize_index(new_size)
            logger.info("Resizing HNSW index.")

        if metadata:
            self._metadata.update(dict(zip(ids, metadata)))
        self._index.add_items(data=embeddings, ids=ids, num_threads=self.config.num_threads)

    async def get_knn(
        self, embeddings: Iterable[GTEmbedding], top_k: int
    ) -> Tuple[Iterable[Iterable[GTId]], npt.NDArray[TScore]]:
        if self.size == 0:
            empty_list: List[List[GTId]] = []
            logger.info("Querying knns in empty index.")
            return empty_list, np.array([], dtype=TScore)

        top_k = min(top_k, self.size)

        if top_k > self.config.ef_search:
            self._index.set_ef(top_k)

        # distances is [0, 2] (best, worst)
        ids, distances = self._index.knn_query(data=embeddings, k=top_k, num_threads=self.config.num_threads)

        return ids, 1.0 - np.array(distances, dtype=TScore) * 0.5

    async def score_all(
        self,
        embeddings: Iterable[GTEmbedding],
        top_k: int = 1,
        threshold: Optional[float] = None,
    ) -> csr_matrix:
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(list(embeddings), dtype=np.float32)

        if embeddings.size == 0 or self.size == 0:
            logger.warning(f"No provided embeddings ({embeddings.size}) or empty index ({self.size}).")
            return csr_matrix((0, self.size))

        top_k = min(top_k, self.size)
        if top_k > self.config.ef_search:
            self._index.set_ef(top_k)

        # distances is [0, 2] (best, worst)
        ids, distances = self._index.knn_query(data=embeddings, k=top_k, num_threads=self.config.num_threads)

        ids = np.array(ids)
        scores = 1.0 - np.array(distances, dtype=TScore) * 0.5

        if threshold is not None:
            scores[scores < threshold] = 0

        # Create sparse distance matrix with shape (#embeddings, #all_embeddings)
        flattened_ids = ids.ravel()
        flattened_scores = scores.ravel()

        scores = csr_matrix(
            (flattened_scores, (np.repeat(np.arange(len(ids)), top_k), flattened_ids)),
            shape=(len(ids), self.size),
        )

        return scores

    async def _insert_start(self):
        self._index = hnswlib.Index(space="cosine", dim=self.embedding_dim)  # type: ignore

        if self.namespace:
            index_file_name = self.namespace.get_load_path(self.RESOURCE_NAME.format(self.embedding_dim))
            metadata_file_name = self.namespace.get_load_path(self.RESOURCE_METADATA_NAME)

            if index_file_name and metadata_file_name:
                try:
                    self._index.load_index(index_file_name, allow_replace_deleted=True)
                    with open(metadata_file_name, "rb") as f:
                        self._metadata = pickle.load(f)
                        logger.debug(f"Loaded {self.size} elements from vectordb storage '{index_file_name}'.")
                    return  # All good
                except Exception as e:
                    t = f"Error loading metadata file for vectordb storage '{metadata_file_name}': {e}"
                    logger.error(t)
                    raise InvalidStorageError(t) from e
            else:
                logger.info(f"No data file found for vectordb storage '{index_file_name}'. Loading empty vectordb.")
        else:
            logger.debug("Creating new volatile vectordb storage.")
        self._index.init_index(
            max_elements=self.INITIAL_MAX_ELEMENTS,
            ef_construction=self.config.ef_construction,
            M=self.config.M,
            allow_replace_deleted=True,
        )
        self._index.set_ef(self.config.ef_search)
        self._metadata = {}

    async def _insert_done(self):
        if self.namespace:
            index_file_name = self.namespace.get_save_path(self.RESOURCE_NAME.format(self.embedding_dim))
            metadata_file_name = self.namespace.get_save_path(self.RESOURCE_METADATA_NAME)

            try:
                self._index.save_index(index_file_name)
                with open(metadata_file_name, "wb") as f:
                    pickle.dump(self._metadata, f)
                logger.debug(f"Saving {self.size} elements from vectordb storage '{index_file_name}'.")
            except Exception as e:
                t = f"Error saving vectordb storage from {index_file_name}: {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e

    async def _query_start(self):
        assert self.namespace, "Loading a vectordb requires a namespace."
        self._index = hnswlib.Index(space="cosine", dim=self.embedding_dim)  # type: ignore

        index_file_name = self.namespace.get_load_path(self.RESOURCE_NAME.format(self.embedding_dim))
        metadata_file_name = self.namespace.get_load_path(self.RESOURCE_METADATA_NAME)
        if index_file_name and metadata_file_name:
            try:
                self._index.load_index(index_file_name, allow_replace_deleted=True)
                with open(metadata_file_name, "rb") as f:
                    self._metadata = pickle.load(f)
                logger.debug(f"Loaded {self.size} elements from vectordb storage '{index_file_name}'.")

                return  # All good
            except Exception as e:
                t = f"Error loading vectordb storage from {index_file_name}: {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e
        else:
            logger.warning(f"No data file found for vectordb storage '{index_file_name}'. Loading empty vectordb.")
        self._index.init_index(
            max_elements=self.INITIAL_MAX_ELEMENTS,
            ef_construction=self.config.ef_construction,
            M=self.config.M,
            allow_replace_deleted=True,
        )
        self._index.set_ef(self.config.ef_search)
        self._metadata = {}

    async def _query_done(self):
        pass

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple, Union

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
    embedding_dim: int = field()
    ef_construction: int = field(default=100)
    M: int = field(default=16)
    max_elements: int = field(default=1000000)
    ef_search: int = field(default=50)
    num_threads: int = field(default=-1)


@dataclass
class HNSWVectorStorage(BaseVectorStorage[GTId, GTEmbedding]):
    RESOURCE_NAME = "hnsw_index_{}.bin"
    RESOURCE_METADATA_NAME = "hnsw_metadata.pkl"
    config: HNSWVectorStorageConfig = field()  # type: ignore
    _index: Any = field(init=False, default=None)  # type: ignore
    _metadata: Dict[GTId, Dict[str, Any]] = field(default_factory=dict)
    _current_elements: int = field(init=False, default=0)

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    async def upsert(
        self,
        ids: Iterable[GTId],
        embeddings: Iterable[GTEmbedding],
        metadata: Union[Iterable[Dict[str, Any]], None] = None,
    ) -> None:
        ids = list(ids)
        embeddings = np.array(list(embeddings), dtype=np.float32)
        metadata = list(metadata) if metadata else None

        assert (len(ids) == len(embeddings)) and (
            metadata is None or (len(metadata) == len(ids))
        ), "ids, embeddings, and metadata (if provided) must have the same length"

        # TODO: this should expand the index
        if self._current_elements + len(embeddings) > self.config.max_elements:
            logger.error(f"HNSW index is full. Cannot insert {len(embeddings)} elements.")
            raise NotImplementedError(f"Cannot insert {len(embeddings)} elements. Full index.")

        if metadata:
            self._metadata.update(dict(zip(ids, metadata)))
        self._index.add_items(data=embeddings, ids=ids, num_threads=self.config.num_threads)
        self._current_elements = self._index.get_current_count()

    async def get_knn(self, embeddings: Iterable[GTEmbedding], top_k: int) -> Tuple[List[GTId], npt.NDArray[TScore]]:
        if self._current_elements == 0:
            empty_list: List[GTId] = []
            logger.info("Querying knns in empty index.")
            return empty_list, np.array([], dtype=TScore)

        top_k = min(top_k, self._current_elements)

        if top_k > self.config.ef_search:
            self._index.set_ef(top_k)

        ids, distances = self._index.knn_query(data=embeddings, k=top_k, num_threads=self.config.num_threads)

        return ids, 1.0 - np.array(distances, dtype=TScore)

    async def score_all(
        self, embeddings: Iterable[GTEmbedding], top_k: int = 1, confidence_threshold: float = 0.0
    ) -> csr_matrix:
        if confidence_threshold > 0.0:
            raise NotImplementedError("Confidence threshold is not supported yet.")
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(list(embeddings), dtype=np.float32)

        if embeddings.size == 0 or self._current_elements == 0:
            logger.warning(
                f"No provided embeddings ({int(embeddings.size / self.embedding_dim)})"
                " or empty index ({self._current_elements})."
            )
            return csr_matrix((0, self._current_elements))

        ids, distances = self._index.knn_query(data=embeddings, k=top_k, num_threads=self.config.num_threads)

        ids = np.array(ids)
        scores = np.array(distances, dtype=TScore)

        # Create sparse distance matrix with shape (#embeddings, #all_embeddings)
        flattened_ids = ids.ravel()
        flattened_scores = scores.ravel()

        scores = csr_matrix(
            (flattened_scores, (np.repeat(np.arange(len(ids)), top_k), flattened_ids)),
            shape=(len(ids), self._current_elements),
        )

        scores.data = (2.0 - scores.data) * 0.5

        return scores

    async def _insert_start(self):
        self._index = hnswlib.Index(space="cosine", dim=self.config.embedding_dim)  # type: ignore

        if self.namespace:
            index_file_name = self.namespace.get_load_path(self.RESOURCE_NAME.format(self.config.embedding_dim))
            metadata_file_name = self.namespace.get_load_path(self.RESOURCE_METADATA_NAME)

            if index_file_name and metadata_file_name:
                try:
                    self._index.load_index(index_file_name, max_elements=self.config.max_elements)
                    with open(metadata_file_name, "rb") as f:
                        self._metadata, self._current_elements = pickle.load(f)
                        logger.debug(
                            f"Loaded {self._current_elements} elements from vectordb storage '{index_file_name}'."
                        )
                except Exception as e:
                    t = f"Error loading metadata file for vectordb storage '{metadata_file_name}': {e}"
                    logger.error(t)
                    raise InvalidStorageError(t) from e
            else:
                logger.info(f"No data file found for vectordb storage '{index_file_name}'. Loading empty vectordb.")
                self._index.init_index(
                    max_elements=self.config.max_elements,
                    ef_construction=self.config.ef_construction,
                    M=self.config.M,
                )
                self._index.set_ef(self.config.ef_search)
                self._metadata = {}
                self._current_elements = 0
        else:
            self._index.init_index(
                max_elements=self.config.max_elements,
                ef_construction=self.config.ef_construction,
                M=self.config.M,
            )
            self._index.set_ef(self.config.ef_search)
            self._metadata = {}
            self._current_elements = 0
            logger.debug("Creating new volatile vectordb storage.")

    async def _insert_done(self):
        if self.namespace:
            index_file_name = self.namespace.get_save_path(self.RESOURCE_NAME.format(self.config.embedding_dim))
            metadata_file_name = self.namespace.get_save_path(self.RESOURCE_METADATA_NAME)

            try:
                self._index.save_index(index_file_name)
                with open(metadata_file_name, "wb") as f:
                    pickle.dump((self._metadata, self._current_elements), f)
                logger.debug(f"Saving {self._current_elements} elements from vectordb storage '{index_file_name}'.")
            except Exception as e:
                t = f"Error saving vectordb storage from {index_file_name}: {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e

    async def _query_start(self):
        assert self.namespace, "Loading a vectordb requires a namespace."
        self._index = hnswlib.Index(space="cosine", dim=self.config.embedding_dim)  # type: ignore

        index_file_name = self.namespace.get_load_path(self.RESOURCE_NAME.format(self.config.embedding_dim))
        metadata_file_name = self.namespace.get_load_path(self.RESOURCE_METADATA_NAME)
        if index_file_name and metadata_file_name:
            try:
                self._index.load_index(index_file_name, max_elements=self.config.max_elements)
                with open(metadata_file_name, "rb") as f:
                    self._metadata, self._current_elements = pickle.load(f)
                logger.debug(f"Loaded {self._current_elements} elements from vectordb storage '{index_file_name}'.")
            except Exception as e:
                t = f"Error loading vectordb storage from {index_file_name}: {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e
        else:
            logger.warning(f"No data file found for vectordb storage '{index_file_name}'. Loading empty vectordb.")
            self._metadata = {}
            self._current_elements = 0

    async def _query_done(self):
        pass

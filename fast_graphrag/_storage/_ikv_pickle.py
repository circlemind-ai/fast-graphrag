import pickle
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import numpy.typing as npt

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._types import GTKey, GTValue, TIndex
from fast_graphrag._utils import logger

from ._base import BaseIndexedKeyValueStorage


@dataclass
class PickleIndexedKeyValueStorage(BaseIndexedKeyValueStorage[GTKey, GTValue]):
    RESOURCE_NAME = "kv_data.pkl"
    _data: Dict[Union[None, TIndex], GTValue] = field(init=False, default_factory=dict)
    _key_to_index: Dict[GTKey, TIndex] = field(init=False, default_factory=dict)
    _free_indices: List[TIndex] = field(init=False, default_factory=list)
    _np_keys: Optional[npt.NDArray[np.object_]] = field(init=False, default=None)

    async def size(self) -> int:
        return len(self._data)

    async def get(self, keys: Iterable[GTKey]) -> Iterable[Optional[GTValue]]:
        return (self._data.get(self._key_to_index.get(key, None), None) for key in keys)

    async def get_by_index(self, indices: Iterable[TIndex]) -> Iterable[Optional[GTValue]]:
        return (self._data.get(index, None) for index in indices)

    async def get_index(self, keys: Iterable[GTKey]) -> Iterable[Optional[TIndex]]:
        return (self._key_to_index.get(key, None) for key in keys)

    async def upsert(self, keys: Iterable[GTKey], values: Iterable[GTValue]) -> None:
        for key, value in zip(keys, values):
            index = self._key_to_index.get(key, None)
            if index is None:
                if len(self._free_indices) > 0:
                    index = self._free_indices.pop()
                else:
                    index = TIndex(len(self._data))
                self._key_to_index[key] = index

                # Invalidate cache
                self._np_keys = None
            self._data[index] = value

    async def delete(self, keys: Iterable[GTKey]) -> None:
        for key in keys:
            index = self._key_to_index.pop(key, None)
            if index is not None:
                self._free_indices.append(index)
                self._data.pop(index, None)

                # Invalidate cache
                self._np_keys = None
            else:
                logger.warning(f"Key '{key}' not found in indexed key-value storage.")

    async def mask_new(self, keys: Iterable[GTKey]) -> Iterable[bool]:
        keys = list(keys)

        if len(keys) == 0:
            return np.array([], dtype=bool)

        if self._np_keys is None:
            self._np_keys = np.fromiter(
                self._key_to_index.keys(),
                count=len(self._key_to_index),
                dtype=type(keys[0]),
            )
        keys_array = np.array(keys, dtype=type(keys[0]))

        return ~np.isin(keys_array, self._np_keys)

    async def _insert_start(self):
        if self.namespace:
            data_file_name = self.namespace.get_load_path(self.RESOURCE_NAME)

            if data_file_name:
                try:
                    print(f"About to save pickle file to {data_file_name}")  # Add this
                    with open(data_file_name, "rb") as f:
                        self._data, self._free_indices, self._key_to_index = pickle.load(f)
                        print(f"Loaded {len(self._data)} elements from indexed key-value storage '{data_file_name}'.")
                    print(f"Successfully saved pickle file to {data_file_name}")  # Add this
                    print(f"Saving {len(self._data)} elements to indexed key-value storage '{data_file_name}'.")
                except Exception as e:
                    t = f"Error loading data file for key-vector storage '{data_file_name}': {e}"
                    logger.error(t)
                    raise InvalidStorageError(t) from e
            else:
                logger.info(f"No data file found for key-vector storage '{data_file_name}'. Loading empty storage.")
                self._data = {}
                self._free_indices = []
                self._key_to_index = {}
        else:
            self._data = {}
            self._free_indices = []
            self._key_to_index = {}
            logger.debug("Creating new volatile indexed key-value storage.")
        self._np_keys = None

    async def _insert_done(self):
        if self.namespace:
            data_file_name = self.namespace.get_save_path(self.RESOURCE_NAME)
            try:
                with open(data_file_name, "wb") as f:
                    pickle.dump((self._data, self._free_indices, self._key_to_index), f)
                    logger.debug(f"Saving {len(self._data)} elements to indexed key-value storage '{data_file_name}'.")
            except Exception as e:
                t = f"Error saving data file for key-vector storage '{data_file_name}': {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e

    async def _query_start(self):
        assert self.namespace, "Loading a kv storage requires a namespace."
        data_file_name = self.namespace.get_load_path(self.RESOURCE_NAME)
        if data_file_name:
            try:
                with open(data_file_name, "rb") as f:
                    self._data, self._free_indices, self._key_to_index = pickle.load(f)
                    logger.debug(
                        f"Loaded {len(self._data)} elements from indexed key-value storage '{data_file_name}'."
                    )
            except Exception as e:
                t = f"Error loading data file for key-vector storage {data_file_name}: {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e
        else:
            logger.warning(f"No data file found for key-vector storage '{data_file_name}'. Loading empty storage.")
            self._data = {}
            self._free_indices = []
            self._key_to_index = {}
        self._np_keys = None

    async def _query_done(self):
        pass

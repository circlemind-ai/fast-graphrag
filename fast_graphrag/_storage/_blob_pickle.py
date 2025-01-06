import pickle
from dataclasses import dataclass, field
from typing import Optional

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._types import GTBlob
from fast_graphrag._utils import logger

from ._base import BaseBlobStorage


@dataclass
class PickleBlobStorage(BaseBlobStorage[GTBlob]):
    RESOURCE_NAME = "blob_data.pkl"
    _data: Optional[GTBlob] = field(init=False, default=None)

    async def get(self) -> Optional[GTBlob]:
        return self._data

    async def set(self, blob: GTBlob) -> None:
        self._data = blob

    async def _insert_start(self):
        if self.namespace:
            data_file_name = self.namespace.get_load_path(self.RESOURCE_NAME)
            if data_file_name:
                try:
                    with open(data_file_name, "rb") as f:
                        self._data = pickle.load(f)
                except Exception as e:
                    t = f"Error loading data file for blob storage {data_file_name}: {e}"
                    logger.error(t)
                    raise InvalidStorageError(t) from e
            else:
                logger.info(f"No data file found for blob storage {data_file_name}. Loading empty storage.")
                self._data = None
        else:
            self._data = None
            logger.debug("Creating new volatile blob storage.")

    async def _insert_done(self):
        if self.namespace:
            data_file_name = self.namespace.get_save_path(self.RESOURCE_NAME)
            try:
                with open(data_file_name, "wb") as f:
                    pickle.dump(self._data, f)
                logger.debug(f"Saving blob storage '{data_file_name}'.")
            except Exception as e:
                logger.error(f"Error saving data file for blob storage {data_file_name}: {e}")

    async def _query_start(self):
        assert self.namespace, "Loading a blob storage requires a namespace."

        data_file_name = self.namespace.get_load_path(self.RESOURCE_NAME)
        if data_file_name:
            try:
                with open(data_file_name, "rb") as f:
                    self._data = pickle.load(f)
            except Exception as e:
                t = f"Error loading data file for blob storage {data_file_name}: {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e
        else:
            logger.warning(f"No data file found for blob storage {data_file_name}. Loading empty blob.")
            self._data = None

    async def _query_done(self):
        pass

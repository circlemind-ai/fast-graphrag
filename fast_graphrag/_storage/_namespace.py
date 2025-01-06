import os
import shutil
import time
from typing import Any, Callable, List, Optional

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._utils import logger


class Workspace:
    @staticmethod
    def new(working_dir: str, checkpoint: int = 0, keep_n: int = 0) -> "Workspace":
        return Workspace(working_dir, checkpoint, keep_n)

    @staticmethod
    def get_path(working_dir: str, checkpoint: Optional[int] = None) -> Optional[str]:
        if checkpoint is None:
            return None
        elif checkpoint == 0:
            return working_dir
        return os.path.join(working_dir, str(checkpoint))

    def __init__(self, working_dir: str, checkpoint: int = 0, keep_n: int = 0):
        self.working_dir: str = working_dir
        self.keep_n: int = keep_n
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        self.checkpoints = sorted(
            (
                int(x.name)
                for x in os.scandir(self.working_dir)
                if x.is_dir() and not x.name.startswith("0__err_") and not x.name.startswith("batch_prompts")
            ),
            reverse=True,
        )
        if self.checkpoints:
            self.current_load_checkpoint = checkpoint if checkpoint else self.checkpoints[0]
        else:
            self.current_load_checkpoint = checkpoint
        self.save_checkpoint: Optional[int] = None
        self.failed_checkpoints: List[str] = []

    def __del__(self):
        for checkpoint in self.failed_checkpoints:
            old_path = os.path.join(self.working_dir, checkpoint)
            new_path = os.path.join(self.working_dir, f"0__err_{checkpoint}")
            os.rename(old_path, new_path)

        if self.keep_n > 0:
            checkpoints = sorted(
                (x.name for x in os.scandir(self.working_dir) if x.is_dir()),
                reverse=True,
            )
            for checkpoint in checkpoints[self.keep_n + 1 :]:
                shutil.rmtree(os.path.join(self.working_dir, str(checkpoint)))

    def make_for(self, namespace: str) -> "Namespace":
        return Namespace(self, namespace)

    def get_load_path(self) -> Optional[str]:
        load_path = self.get_path(self.working_dir, self.current_load_checkpoint)
        if load_path == self.working_dir and len([x for x in os.scandir(load_path) if x.is_file()]) == 0:
            return None
        return load_path

    def get_save_path(self) -> str:
        if self.save_checkpoint is None:
            if self.keep_n > 0:
                self.save_checkpoint = int(time.time())
            else:
                self.save_checkpoint = 0
        save_path = self.get_path(self.working_dir, self.save_checkpoint)

        assert save_path is not None, "Save path cannot be None."

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return os.path.join(save_path)

    def _rollback(self) -> bool:
        if self.current_load_checkpoint is None:
            return False
        # List all directories in the working directory and select the one
        # with the smallest number greater then the current load checkpoint.
        try:
            self.current_load_checkpoint = next(x for x in self.checkpoints if x < self.current_load_checkpoint)
            logger.warning("Rolling back to checkpoint: %s", self.current_load_checkpoint)
        except (StopIteration, ValueError):
            self.current_load_checkpoint = None
            logger.warning(
                "No checkpoints to rollback to. Last checkpoint tried: %s",
                self.current_load_checkpoint,
            )

        return True

    async def with_checkpoints(self, fn: Callable[[], Any]) -> Any:
        while True:
            try:
                return await fn()
            except Exception as e:
                logger.warning("Error occurred loading checkpoint: %s", e)
                if self.current_load_checkpoint is not None:
                    self.failed_checkpoints.append(str(self.current_load_checkpoint))
                if self._rollback() is False:
                    break
        raise InvalidStorageError("No valid checkpoints to load or default storages cannot be created.")


class Namespace:
    def __init__(self, workspace: Workspace, namespace: Optional[str] = None):
        self.namespace = namespace
        self.workspace = workspace

    def get_load_path(self, resource_name: str) -> Optional[str]:
        assert self.namespace is not None, "Namespace must be set to get resource load path."
        load_path = self.workspace.get_load_path()
        if load_path is None:
            return None
        return os.path.join(load_path, f"{self.namespace}_{resource_name}")

    def get_save_path(self, resource_name: str) -> str:
        assert self.namespace is not None, "Namespace must be set to get resource save path."
        return os.path.join(self.workspace.get_save_path(), f"{self.namespace}_{resource_name}")

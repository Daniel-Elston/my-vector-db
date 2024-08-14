# from __future__ import annotations

# from pathlib import Path
# from typing import Callable
# from typing import List
# from typing import Optional
# from typing import Union

# import pandas as pd
# from config.state_init import StateManager
# from utils.file_access import FileAccess
# from utils.logging_utils import log_step


# class TaskExecutor:
#     def __init__(self, state: StateManager):
#         self.data_config = state.data_config
#         self.paths = state.paths

#     def run_main_step(
#             self, step: Callable,
#             load_path: Optional[Union[str, List[str], Path]] = None,
#             save_paths: Optional[Union[str, List[str], Path]] = None,
#             args: Optional[Union[dict, None]] = None,
#             kwargs: Optional[Union[dict, None]] = None) -> pd.DataFrame:
#         """Pipeline runner for top-level main.py."""
#         return step(**args) if args is not None else step()

#     def run_parent_step(
#             self, step: Callable,
#             load_path: Optional[Union[str, List[str], Path]] = None,
#             save_paths: Optional[Union[str, List[str], Path]] = None,
#             df: Optional[Union[pd.DataFrame]] = None):
#         """Pipeline runner for parent pipelines scripts (src/pipelines/*)"""
#         if load_path is None:
#             self._parent_save_helper(step, df, load_path, save_paths)

#         else:
#             load_path = self.paths.get_path(load_path)
#             with FileAccess.load_file(load_path) as df:
#                 self._parent_save_helper(step, df, load_path, save_paths)

#     def _parent_save_helper(self, step, df, load_path, save_paths):
#         if save_paths is not None:
#             if isinstance(save_paths, str):
#                 save_paths = self.paths.get_path(save_paths)
#                 logged_step = log_step(load_path, save_paths)(step)
#                 result = logged_step(df)
#                 FileAccess.save_file(result, save_paths, self.data_config.overwrite)
#             if isinstance(save_paths, list):
#                 save_paths = [self.paths.get_path(path) for path in save_paths]
#                 for path in save_paths:
#                     logged_step = log_step(load_path, save_paths)(step)
#                     result = logged_step(df)
#                     FileAccess.save_file(result, path, self.data_config.overwrite)
#         else:
#             logged_step = log_step(load_path, save_paths)(step)
#             result = logged_step(df)
#             return result

#     @staticmethod
#     def run_child_step(
#             step: Callable,
#             df: pd.DataFrame,
#             args: Optional[Union[dict, None]] = None,
#             kwargs: Optional[Union[dict, None]] = None) -> pd.DataFrame:
#         """Pipeline runner for child pipelines scripts (lowest level scripts)"""
#         return step(df, **args) if args is not None else step(df)

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Union

import pandas as pd

from config.state_init import StateManager
from utils.file_access import FileAccess
from utils.logging_utils import log_step


class TaskExecutor:
    def __init__(self, state: StateManager):
        self.data_config = state.data_config
        self.paths = state.paths

    def run_main_step(
        self,
        step: Callable,
        load_path: Optional[Union[str, List[str], Path]] = None,
        save_paths: Optional[Union[str, List[str], Path]] = None,
        args: Optional[Union[dict, None]] = None,
        kwargs: Optional[Union[dict, None]] = None,
    ) -> pd.DataFrame:
        """Pipeline runner for top-level main.py."""
        return step() if args is None else step(**args)

    def run_parent_step(
        self,
        step: Callable,
        load_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        save_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        df: Optional[Union[pd.DataFrame]] = None,
    ) -> Optional[pd.DataFrame]:
        """Pipeline runner for parent pipelines scripts (src/pipelines/*)"""
        if load_path is not None:
            if isinstance(load_path, (str, Path)):
                load_path = self.paths.get_path(load_path)
                with FileAccess.load_file(load_path) as df:
                    logged_step = log_step(load_path, save_paths)(step)
                    result = logged_step(df)
                    return result
            else:
                pass

        if save_paths is not None:
            if isinstance(save_paths, (str, Path)):
                save_paths = [self.paths.get_path(save_paths)]
            else:
                save_paths = [self.paths.get_path(path) for path in save_paths]

            logged_step = log_step(load_path, save_paths)(step)
            result = logged_step()

            for path in save_paths:
                FileAccess.save_file(result, path, self.data_config.overwrite)
        return result

    @staticmethod
    def run_child_step(
        step: Callable,
        df: pd.DataFrame,
        df_response: Optional[pd.DataFrame] = None,
        args: Optional[Union[dict, None]] = None,
        kwargs: Optional[Union[dict, None]] = None,
    ) -> pd.DataFrame:
        """Pipeline runner for child pipelines scripts (lowest level scripts)"""
        try:
            return (
                step(df, df_response=df_response, **kwargs)
                if kwargs is not None
                else step(df, df_response=df_response)
            )
        except TypeError:
            return step(df, **kwargs) if kwargs is not None else step(df)

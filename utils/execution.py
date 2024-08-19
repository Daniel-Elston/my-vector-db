from __future__ import annotations

import logging
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
    ) -> pd.DataFrame:
        """Pipeline runner for top-level main.py."""
        return step() if args is None else step(**args)

    def run_parent_step(
        self,
        step: Callable,
        load_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        save_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        df: Optional[Union[pd.DataFrame]] = None,
        *args,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Pipeline runner for parent pipelines scripts (src/pipelines/*)"""
        if load_path is not None:
            if isinstance(load_path, (str, Path)):
                load_path = self.paths.get_path(load_path)
                with FileAccess.load_file(load_path) as df:
                    df = df
            else:
                pass

        if save_paths is not None:
            if isinstance(save_paths, (str, Path)):
                save_paths = [self.paths.get_path(save_paths)]
            if isinstance(save_paths, list):
                save_paths = [self.paths.get_path(path) for path in save_paths]
        if save_paths is None:
            logged_step = log_step(load_path, save_paths)(step)
            result = (
                logged_step(df, *args, **kwargs)
                if df is not None
                else logged_step(*args, **kwargs)
            )
            return result

        logged_step = log_step(load_path, save_paths)(step)
        result = (
            logged_step(df, *args, **kwargs) if df is not None else logged_step(*args, **kwargs)
        )

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

    def _execute_steps(self, steps, stage=None):
        if stage == "main":
            for step, load_path, save_paths in steps:
                logging.info(
                    f"INITIATING {step.__self__.__class__.__name__} with:\n"
                    f"    Input_path: {self.paths.get_path(load_path)}\n"
                    f"    Output_paths: {self.paths.get_path(save_paths)}\n"
                )
                self.run_main_step(step, load_path, save_paths)
                logging.info(f"{step.__self__.__class__.__name__} completed SUCCESSFULLY.\n")
        if stage == "parent":
            for step, load_path, save_paths in steps:
                self.run_parent_step(step, load_path, save_paths)

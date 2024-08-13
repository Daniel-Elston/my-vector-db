from __future__ import annotations

import pandas as pd

from config.state_init import StateManager
from utils.execution import TaskExecutor


class MakeDataset:
    """Load dataset and perform base processing"""

    def __init__(self, state: StateManager):
        self.state = state

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        steps = [
            self.base_process,
        ]
        for step in steps:
            df = TaskExecutor.run_child_step(step, df)
        return df

    def base_process(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {"Unnamed: 0": "uid"}
        df = df.rename(columns=rename_map)
        return df.drop(columns=["key"])

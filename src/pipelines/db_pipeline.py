from __future__ import annotations

import pandas as pd

from config.state_init import StateManager
from utils.execution import TaskExecutor


class DatabasePipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor, stage: str):
        self.state = state
        self.exe = exe
        self.stage = stage
        self._set_paths_and_table()

    def insert_load(self):
        steps = [
            (self._create_table, self.load_path, None),
            (self._insert_data, self.load_path, None),
            (self._fetch_data, None, self.save_paths),
        ]
        self.exe._execute_steps(steps, stage="parent")

    def _set_paths_and_table(self):
        if self.stage == "raw":
            self.load_path = "raw"
            self.save_paths = "load_raw"
            self.table_name = "raw_docs"
        elif self.stage == "vectorised":
            self.load_path = "vectorised"
            self.save_paths = "load_vector"
            self.table_name = "vector_docs"
        else:
            raise ValueError(f"Invalid stage: {self.stage}")

    def _create_table(self, df: pd.DataFrame):
        db_ops = self.state.db_manager.ops
        db_ops.table = self.table_name
        db_ops.create_table_if_not_exists(df)
        return df

    def _insert_data(self, df: pd.DataFrame):
        data_handler = self.state.db_manager.handler
        data_handler.table = self.table_name
        data_handler.insert_batches_to_db(df)
        return df

    def _fetch_data(self):
        data_handler = self.state.db_manager.handler
        data_handler.table = self.table_name
        query = f"SELECT * FROM {self.state.db_manager.config.schema}.{self.table_name};"
        return data_handler.fetch_data(query)

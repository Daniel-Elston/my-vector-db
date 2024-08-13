from __future__ import annotations

from config.state_init import StateManager
from utils.execution import TaskExecutor


class InsertPipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe

    def main(self):
        steps = [
            (self.state.db_manager.ops.create_table_if_not_exists, "raw", None),
            (self.state.db_manager.handler.insert_batches_to_db, "raw", None),
        ]
        for step, load_path, save_paths in steps:
            self.exe.run_parent_step(step, load_path, save_paths)


class FetchPipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe

    def main(self):
        steps = [
            (self.state.db_manager.handler.fetch_data, None, "load"),
        ]
        for step, load_path, save_paths in steps:
            self.exe.run_parent_step(step, load_path, save_paths)

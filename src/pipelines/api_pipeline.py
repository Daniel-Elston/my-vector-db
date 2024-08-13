from __future__ import annotations

from config.state_init import StateManager
from src.api.insert import InsertData
from src.api.request import RequestData
from utils.execution import TaskExecutor
from utils.logging_utils import log_cls_methods


@log_cls_methods
class RequestPipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe

    def main(self):
        steps = [
            (RequestData(self.state).make_request, None, None),
            (InsertData(self.state).pipeline, "response", None),
        ]
        for step, load_path, save_paths in steps:
            self.exe.run_parent_step(step, load_path, save_paths)

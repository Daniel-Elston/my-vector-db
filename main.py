from __future__ import annotations

import logging

from config.state_init import StateManager
from src.pipelines.data_pipeline import DataPipeline
from src.pipelines.db_pipeline import FetchPipeline, InsertPipeline
from utils.execution import TaskExecutor
from utils.project_setup import setup_project

# from src.pipelines.api_pipeline import RequestPipeline


class MainPipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe

    def run(self):
        logging.info(
            f"INITIATING {__class__.__name__} from top-level script: ``{__file__.split('/')[-1]}``...\n"
        )
        steps = [
            (DataPipeline(self.state, self.exe).main, "raw", "sdo"),
            (InsertPipeline(self.state, self.exe).main, "raw", None),
            (FetchPipeline(self.state, self.exe).main, None, "load"),
            # (RequestPipeline(self.state, self.exe), 'main', 'raw', 'sdo'),
        ]
        for step, load_path, save_paths in steps:
            logging.info(
                f"INITIATING {step.__self__.__class__.__name__} with:\n"
                f"    Input_path: {self.state.paths.get_path(load_path)}\n"
                f"    Output_paths: {self.state.paths.get_path(save_paths)}\n"
            )
            self.exe.run_main_step(step, load_path, save_paths)
            logging.info(f"{step.__self__.__class__.__name__} completed SUCCESSFULLY.\n")
        logging.info(
            f'Completed {__class__.__name__} from top-level script: ``{__file__.split("/")[-1]}`` SUCCESSFULLY.\n'
        )


if __name__ == "__main__":
    project_dir, project_config, state_manager = setup_project()
    exe = TaskExecutor(state_manager)

    try:
        MainPipeline(state_manager, exe).run()
    except Exception as e:
        logging.error(f"Pipeline terminated due to unexpected error: {e}", exc_info=True)

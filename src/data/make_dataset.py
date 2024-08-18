from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from config.state_init import StateManager
from utils.execution import TaskExecutor


class MakeDataset:
    """Load dataset and perform base processing"""

    def __init__(self, state: StateManager):
        self.state = state
        self.dc = state.data_config

    def pipeline(self) -> pd.DataFrame:
        df = self.make_raw_set()

        steps = [
            self.add_document,
        ]
        for step in steps:
            df = TaskExecutor.run_child_step(step, df)
        return df

    def make_raw_set(self):
        docs_path = self.state.paths.get_path("raw")
        with open(docs_path, "r") as file:
            stories = json.load(file)
        return pd.DataFrame(stories)

    def add_document(self, df):
        """Add a new document to the DataFrame."""
        if self.dc.input_title and self.dc.input_document is not None:
            new_id = df["id"].max() + 1 if not df.empty else 1
            new_row = {
                "id": new_id,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "title": self.dc.input_title,
                "document": self.dc.input_document,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df = df.drop_duplicates(subset=["title", "document"], keep="last")
            return df
        else:
            return df

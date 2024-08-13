from __future__ import annotations

import gc
import logging
import time

import pandas as pd

from config.state_init import StateManager
from utils.execution import TaskExecutor


class InsertData:
    def __init__(self, state: StateManager):
        self.db_config = state.db_manager.config
        self.api_config = state.api_config
        self.db_ops = state.db_manager.ops
        self.handler = state.db_manager.handler

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        steps = [self.normalize_data, self.split_to_chunks, self.insert_data, self.cleanup]
        for step in steps:
            df = TaskExecutor.run_child_step(step, df)
        return df

    def normalize_data(self, data_store):
        records = []
        for sensor_id, readings in data_store.items():
            for reading in readings:
                # 'ts': pd.to_datetime(reading['ts'], unit='ms')
                record = {"sensor_id": sensor_id, "ts": reading["ts"], "v": reading["v"]}
                records.append(record)
        return pd.DataFrame(records)

    def split_to_chunks(self, df):
        chunks = [
            df[i: i + self.db_config.chunk_size]
            for i in range(0, df.shape[0], self.db_config.chunk_size)
        ]
        return chunks

    def insert_data(self, df):
        chunks = self.split_to_chunks(df)
        for chunk in chunks:
            self.db_ops.create_table_if_not_exists(chunk)
            self.handler.insert_batches_to_db(chunk)

    def cleanup(self):
        gc.collect()
        logging.info(f"Sleeping for Interval: {self.api_config.sleep_interval} seconds...")
        time.sleep(int(self.api_config.sleep_interval))

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from config.state_init import StateManager
from utils.execution import TaskExecutor


class SimilarityPipeline:
    def __init__(self, state: StateManager):
        self.state = state
        self.dc = state.data_config

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        steps = [
            self.vector_similarity_search,
        ]
        for step in steps:
            df = TaskExecutor.run_child_step(step, df)
        return df

    def vector_similarity_search(self, df: pd.DataFrame) -> pd.DataFrame:
        input_title = self.dc.input_title
        input_vector = df[df["title"] == input_title]["document_vector"].iloc[0]

        input_vector = self.string_to_array(input_vector)

        df["similarity"] = df["document_vector"].apply(
            lambda x: 1 - cosine(input_vector, self.string_to_array(x))
        )

        results = df.sort_values("similarity", ascending=False).head(5)
        results = results[["id", "title", "document", "similarity"]]
        return results

    @staticmethod
    def string_to_array(vector_string):
        if isinstance(vector_string, str):
            vector_string = vector_string.strip("[]")
            vector_list = [float(x) for x in re.split(r"\s+", vector_string) if x]
            return np.array(vector_list)
        elif isinstance(vector_string, np.ndarray):
            return vector_string
        else:
            raise ValueError(f"Unexpected type for vector: {type(vector_string)}")

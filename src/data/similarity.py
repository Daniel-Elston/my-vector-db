from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config.state_init import StateManager
from src.data.embeddings import DocumentEmbeddings
from utils.execution import TaskExecutor


class SimilarityPipeline:
    def __init__(self, state: StateManager, embeddings_model: DocumentEmbeddings):
        self.state = state
        self.dc = state.data_config
        self.embeddings = embeddings_model

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        steps = [
            self.convert_vectors,
            self.vector_similarity_search,
        ]
        for step in steps:
            df = TaskExecutor.run_child_step(step, df)
        return df

    def convert_vectors(self, df: pd.DataFrame) -> pd.DataFrame:
        df["document_vector"] = df["document_vector"].apply(self.safe_convert)
        df = df.dropna(subset=["document_vector"])
        return df

    def vector_similarity_search(self, df: pd.DataFrame) -> pd.DataFrame:
        input_title = self.dc.input_title
        input_document = self.dc.input_document
        input_text = f"{input_title} {input_document}"
        input_vector = self.embeddings.get_document_vector(input_text)
        similarities = cosine_similarity([input_vector], df["document_vector"].tolist())[0]
        df["similarity"] = similarities
        results = df.sort_values("similarity", ascending=False).head(5)
        results = results[["id", "title", "document", "similarity"]]

        return results

    @staticmethod
    def string_to_array(vector_string):
        if isinstance(vector_string, str):
            vector_string = vector_string.strip("[]")
            vector_list = [float(x) for x in re.split(r"\s+", vector_string) if x.strip()]
            return np.array(vector_list)
        elif isinstance(vector_string, np.ndarray):
            return vector_string
        else:
            raise ValueError(f"Unexpected type for vector: {type(vector_string)}")

    def safe_convert(self, vector_string):
        try:
            return self.string_to_array(vector_string)
        except ValueError as e:
            print(f"Error converting vector: {e}")
            print(f"Problematic vector string: {vector_string[:100]}...")
            return None

from __future__ import annotations

import ast
from typing import List

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
        df["document_vector"] = df["document_vector"].apply(self.parse_vector_string)
        return df

    @staticmethod
    def parse_vector_string(vector_string: str) -> List[float]:
        try:
            return ast.literal_eval(vector_string)
        except (ValueError, SyntaxError):
            return []

    def vector_similarity_search(self, df: pd.DataFrame) -> pd.DataFrame:
        input_title = self.dc.input_title
        input_document = self.dc.input_document
        input_text = f"{input_title} {input_document}"

        input_vector = self.embeddings.get_document_vector(input_text)
        document_vectors = np.array(df["document_vector"].tolist())
        similarities = cosine_similarity([input_vector], document_vectors)[0]
        df["similarity"] = similarities
        results = df.sort_values("similarity", ascending=False).head(5)
        results = results[["id", "title", "document", "similarity"]]
        return results

    def get_combined_vector(self, title: str, document: str) -> np.ndarray:
        combined_text = f"{title} {document}"
        return self.embeddings.get_document_vector(combined_text)

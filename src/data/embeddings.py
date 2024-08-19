from __future__ import annotations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config.state_init import StateManager
from utils.execution import TaskExecutor


class DocumentEmbeddings:
    def __init__(self, state: StateManager):
        self.state = state
        self.dc = state.data_config
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        steps = [
            self.create_document_vectors,
        ]
        for step in steps:
            df = TaskExecutor.run_child_step(step, df)
        return df

    def get_document_vector(self, document: str) -> np.ndarray:
        return self.model.encode(document)

    def create_document_vectors(self, df: pd.DataFrame) -> pd.DataFrame:
        documents = df["lemmatized_text"].tolist()
        embeddings = self.model.encode(documents, show_progress_bar=True)
        df["document_vector"] = embeddings.tolist()
        return df

    def get_word_vector(self, word: str) -> np.ndarray:
        return self.model.encode([word])[0]

from __future__ import annotations

import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

from config.state_init import StateManager
from utils.execution import TaskExecutor

nltk.download("punkt_tab")


class DocumentEmbeddings:
    def __init__(self, state: StateManager):
        self.state = state
        self.dc = state.data_config
        # self.vector_size = 300
        # self.window = 5
        # self.min_count = 1
        self.model = None

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        steps = [
            self.train_model,
            self.create_document_vectors,
        ]
        for step in steps:
            df = TaskExecutor.run_child_step(step, df)
        return df

    def train_model(self, df: pd.DataFrame) -> pd.DataFrame:
        documents = df["lemmatized_text"].tolist()
        tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
        self.model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=self.dc.vector_size,
            window=self.dc.window,
            min_count=self.dc.min_count,
        )
        return df

    def get_document_vector(self, document: str) -> np.ndarray:
        words = word_tokenize(document.lower())
        doc_vector = np.mean(
            [self.model.wv[word] for word in words if word in self.model.wv], axis=0
        )
        return doc_vector if doc_vector.size else np.zeros(self.dc.vector_size)

    def create_document_vectors(self, df: pd.DataFrame) -> pd.DataFrame:
        df["document_vector"] = df["lemmatized_text"].apply(self.get_document_vector)
        return df

    def get_word_vector(self, word: str) -> np.ndarray:
        return self.model.wv[word] if word in self.model.wv else None

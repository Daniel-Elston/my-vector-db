from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.tokenize import word_tokenize

from config.state_init import StateManager
from utils.execution import TaskExecutor


class DocumentEmbeddings:
    def __init__(self, state: StateManager):
        self.state = state
        self.dc = state.data_config
        self.model = None

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        steps = [
            self.load_pretrained_model,
            self.create_document_vectors,
        ]
        for step in steps:
            df = TaskExecutor.run_child_step(step, df)
        return df

    def load_pretrained_model(self, df: pd.DataFrame) -> pd.DataFrame:
        glove_path = self.state.paths.get_path("models")
        word2vec_output_file = self.state.paths.get_path("models_output")

        if not os.path.exists(word2vec_output_file):
            logging.info("Converting GloVe to Word2Vec format...")
            glove2word2vec(glove_path, word2vec_output_file)

        logging.info("Loading pre-trained GloVe model...")
        self.model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        return df

    def get_document_vector(self, document: str) -> np.ndarray:
        words = word_tokenize(document.lower())
        word_vectors = [self.model[word] for word in words if word in self.model]
        if not word_vectors:
            return np.zeros(self.model.vector_size)
        return np.mean(word_vectors, axis=0)

    def create_document_vectors(self, df: pd.DataFrame) -> pd.DataFrame:
        df["document_vector"] = df["lemmatized_text"].apply(self.get_document_vector)
        return df

    def get_word_vector(self, word: str) -> np.ndarray:
        return self.model[word] if word in self.model else None

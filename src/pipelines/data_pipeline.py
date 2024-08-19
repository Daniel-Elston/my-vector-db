from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import spacy

from config.state_init import StateManager
from src.data.embeddings import DocumentEmbeddings
from src.data.make_dataset import MakeDataset
from src.data.process import Preprocessor
from src.data.similarity import SimilarityPipeline
from utils.execution import TaskExecutor


class DataPipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe
        self.nlp = spacy.load("en_core_web_sm")
        self.embeddings_model: Optional[DocumentEmbeddings] = None

    def make_raw(self):
        steps = [
            (MakeDataset(self.state), None, "raw"),
        ]
        for step, load_path, save_paths in steps:
            self.exe.run_parent_step(step.pipeline, load_path, save_paths)

    def vectorisation(self):
        self.embeddings_model = DocumentEmbeddings(self.state)
        steps = [
            (Preprocessor(self.state, self.nlp), "load_raw", "process"),
            (self.embeddings_model, "process", "vectorised"),
        ]
        for step, load_path, save_paths in steps:
            self.exe.run_parent_step(step.pipeline, load_path, save_paths)

    def run_vec_sim_search(self):
        if self.embeddings_model is None:
            raise ValueError("Embeddings model not initialized. Run vectorisation first.")

        similarity_pipeline = SimilarityPipeline(self.state, self.embeddings_model)
        steps = [
            (similarity_pipeline.pipeline, "load_vector", "results"),
        ]
        for step, load_path, save_paths in steps:
            self.exe.run_parent_step(step, load_path, save_paths)

        results = pd.read_json(self.state.paths.get_path("results"))
        logging.info(f"RESULTS:\n {results}")
        return results

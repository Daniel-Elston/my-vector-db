from __future__ import annotations

import spacy

from config.state_init import StateManager
from src.data.embeddings import DocumentEmbeddings
from src.data.make_dataset import MakeDataset
from src.data.process import Preprocessor
from utils.execution import TaskExecutor


class DataPipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe
        self.nlp = spacy.load("en_core_web_sm")

    def make_raw(self):
        steps = [
            (MakeDataset(self.state), None, "raw"),
        ]
        for step, load_path, save_paths in steps:
            self.exe.run_parent_step(step.pipeline, load_path, save_paths)

    def vectorisation(self):
        steps = [
            (Preprocessor(self.state, self.nlp), "load_raw", "process"),
            (DocumentEmbeddings(self.state), "process", "vectorised"),
        ]
        for step, load_path, save_paths in steps:
            self.exe.run_parent_step(step.pipeline, load_path, save_paths)

from __future__ import annotations

import re

import contractions
import pandas as pd

from config.state_init import StateManager
from utils.execution import TaskExecutor


class Preprocessor:
    def __init__(self, state: StateManager, nlp):
        self.state = state
        self.dc = state.data_config
        self.nlp = nlp

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        steps = [
            self.clean_text,
            self.apply_lemmatizer,
        ]
        for step in steps:
            df = TaskExecutor.run_child_step(step, df)
        return df

    def clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.dc.text_col] = df[self.dc.text_col].apply(self._clean_text)
        return df

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s*http?://\S+(\s+|$)", " ", text)
        text = re.sub(r"[^\w\s!?#@]", "", text)
        text = contractions.fix(text)
        text = " ".join(text.split())
        return text

    def lemmatize_text(self, text: str) -> str:
        doc = self.nlp(text)
        lemmatized_tokens = [token.lemma_ for token in doc]
        return " ".join(lemmatized_tokens)

    def apply_lemmatizer(self, df: pd.DataFrame) -> pd.DataFrame:
        df["lemmatized_text"] = df[self.dc.text_col].apply(self.lemmatize_text)
        return df

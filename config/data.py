from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    overwrite: bool = True
    save_fig: bool = True
    input_title: Optional[str] = "Cosmic Voyager test"
    input_document: Optional[str] = (
        "The starship Nebula pierced the void, its quantum engines humming"
    )
    text_col: str = "document"
    lemmatized_text_col: str = "lemmatized_text"

    vector_size: int = 300
    window: int = 5
    min_count: int = 1

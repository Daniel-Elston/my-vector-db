from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    overwrite: bool = True
    save_fig: bool = True
    input_title: Optional[str] = None
    input_document: Optional[str] = None
    text_col: str = "document"

    vector_size: int = 300
    window: int = 5
    min_count: int = 1

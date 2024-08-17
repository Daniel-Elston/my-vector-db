from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

paths_store = {
    "raw": "data/raw/raw.parquet",
    "load_raw": "data/db/load/raw-docs.xlsx",
    "process": "data/processed/process.parquet",
    "vectorised": "data/sdo/vectorised.parquet",
    "load_vector": "data/db/load/vectorised.xlsx",
}


@dataclass
class PathsConfig:
    config_path: Path = Path("config/paths.yaml")
    paths: Dict[str, Path] = field(default_factory=dict)

    def __post_init__(self):
        self.paths = {k: Path(v) for k, v in paths_store.items()}

    def get_path(self, key: Optional[Union[str, Path]]) -> Optional[Path]:
        if key is None:
            return None
        if isinstance(key, Path):
            return key
        return self.paths.get(key)

    def validate_paths(self):
        for name, path in self.paths.items():
            if not path.exists():
                logging.warning(f"Path {name} does not exist: {path}")

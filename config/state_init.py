from __future__ import annotations

from dataclasses import dataclass, field

from config.api import ApiConfig
from config.data import DataConfig
from config.db import DatabaseConfig, DatabaseConnManager
from config.paths import PathsConfig


@dataclass
class StateManager:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    db_config: DatabaseConfig = field(default_factory=DatabaseConfig)
    api_config: ApiConfig = field(default_factory=ApiConfig)

    db_manager: DatabaseConnManager = field(init=False)

    def __post_init__(self):
        self.initialize_database()
        self.validate_paths()

    def initialize_database(self):
        self.db_manager = DatabaseConnManager(self.db_config)

    def validate_paths(self):
        self.paths.validate_paths()

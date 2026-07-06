# FastF1 Deps
from fastf1 import Cache

# Core Deps
from pathlib import Path


class DataSetup:
    """This class is reponsible for creating and providing the setup necessary
    to run all of the downstream API."""

    def __init__(self, cache_path: str | None = None, offline_mode: bool = False) -> None:
        # Resolving the Root Paths
        self.root_path = Path()
        self.f1_path = self.root_path / "F1"
        
        # Resolving the Cache Path
        if cache_path:
            self.cache_path = Path(cache_path)
            self.cache_path.mkdir(exist_ok=True)
        else:
            self.cache_path = self.f1_path / "cache"
            self.cache_path.mkdir()

        # Enabling the Cache Directory
        Cache.enable_cache(cache_dir=str(self.cache_path))

        # Use offline mode to rely on the Cache
        if offline_mode:
            Cache.offline_mode(enabled=True)

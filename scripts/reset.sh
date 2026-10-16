"""Wipe the index and rebuild it. Owner: Person 1."""

import shutil

from src.ingest import build_index
from src.settings import INDEX_DIR

if __name__ == "__main__":
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    build_index(rebuild=True)

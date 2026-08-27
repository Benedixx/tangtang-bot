from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from filelock import FileLock

LOGGER = logging.getLogger("tang.storage.json_store")

DEFAULT_DATA_DIR = "./data"


class JsonStore:
    """Atomic JSON read/write with file locks.

    Key rules:
    - Atomic writes only: write to *.tmp, then os.replace() — never write the target file directly.
    - File locking via filelock around read-modify-write cycles.
    - One directory per data type: buffers/, summaries/, facts/, archive/.
    - Per-entity file, not one giant file.
    """

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR) -> None:
        self._root = Path(data_dir)
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for subdir in ("buffers", "summaries", "facts", "archive"):
            (self._root / subdir).mkdir(parents=True, exist_ok=True)

    def _lock_path(self, path: Path) -> Path:
        return path.with_suffix(path.suffix + ".lock")

    def read_json(self, path: Path, default: Any = None) -> Any:
        """Read JSON file with shared lock. Returns default if file missing or corrupt."""
        if not path.exists():
            return default if default is not None else {}

        lock = FileLock(self._lock_path(path))
        with lock:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data
            except (json.JSONDecodeError, OSError):
                LOGGER.exception("json_read_failed path=%s", path)
                quarantine = path.with_suffix(f".corrupt-{int(os.path.getmtime(path))}")
                try:
                    path.rename(quarantine)
                    LOGGER.warning("corrupt_file_quarantined path=%s -> %s", path, quarantine)
                except OSError:
                    pass
                return default if default is not None else {}

    def write_json(self, path: Path, data: Any) -> bool:
        """Atomic write: write to temp file, then os.replace(). Returns True on success."""
        lock = FileLock(self._lock_path(path))
        with lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp_path = tempfile.mkstemp(
                    dir=path.parent,
                    suffix=".tmp",
                    prefix=path.stem,
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=1)
                    os.replace(tmp_path, path)
                    return True
                except OSError:
                    LOGGER.exception("json_write_failed path=%s", path)
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    return False
            except Exception:
                LOGGER.exception("json_write_lock_failed path=%s", path)
                return False

    def append_jsonl(self, path: Path, record: dict[str, Any]) -> bool:
        """Append a JSON line to a .jsonl file (archive). Uses file lock."""
        lock = FileLock(self._lock_path(path))
        with lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                return True
            except OSError:
                LOGGER.exception("jsonl_append_failed path=%s", path)
                return False

    def read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        """Read all lines from a .jsonl file."""
        if not path.exists():
            return []
        lock = FileLock(self._lock_path(path))
        with lock:
            records = []
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            except (json.JSONDecodeError, OSError):
                LOGGER.exception("jsonl_read_failed path=%s", path)
            return records

    def ensure_dirs(self) -> None:
        """Public: create data directories if missing (startup migration)."""
        self._ensure_dirs()

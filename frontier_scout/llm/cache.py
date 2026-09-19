"""On-disk cache for LLM calls.

Keyed by agent + prompt version + input, so editing a prompt invalidates only
that agent's entries. Re-running discovery on the same day should cost nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CallCache:
    def __init__(self, directory: Path, enabled: bool = True):
        self.dir = Path(directory)
        self.enabled = enabled
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        # Shard by prefix; a flat directory of thousands of files gets slow.
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            self.misses += 1
            return None
        self.hits += 1
        return payload

    def put(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, default=str))
        tmp.replace(path)

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_immutable_atomic(path: Path, payload: bytes) -> None:
    """Atomically create immutable content or verify an identical existing file."""
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite immutable artifact: {path.name}.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

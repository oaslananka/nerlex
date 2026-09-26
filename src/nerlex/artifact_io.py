from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_immutable_bytes(path: str | Path, payload: bytes) -> Path:
    """Atomically create an immutable file or verify an identical existing file."""
    destination = Path(path)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError(
                f"Refusing to overwrite immutable artifact file: {destination.name}."
            )
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        if destination.exists():
            if destination.read_bytes() != payload:
                raise ValueError(
                    f"Refusing to overwrite immutable artifact file: {destination.name}."
                )
            return destination

        os.replace(temporary_path, destination)
        return destination
    finally:
        temporary_path.unlink(missing_ok=True)

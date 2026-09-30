from __future__ import annotations

import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path


class DistributionCheckError(RuntimeError):
    """Raised when a built distribution is incomplete or inconsistent."""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dist = root / "dist"
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    name = str(project["name"])
    version = str(project["version"])

    wheel = _one(dist.glob(f"{name}-{version}-*.whl"), "wheel")
    sdist = _one(dist.glob(f"{name}-{version}.tar.gz"), "source distribution")

    _check_wheel(wheel, name=name, version=version)
    _check_sdist(sdist, name=name, version=version)

    print(f"distribution check passed: {wheel.name}, {sdist.name}")


def _one(paths: object, label: str) -> Path:
    matches = tuple(paths)  # type: ignore[arg-type]
    if len(matches) != 1:
        raise DistributionCheckError(
            f"expected exactly one {label}, found {len(matches)}"
        )
    path = matches[0]
    if not isinstance(path, Path):
        raise DistributionCheckError(f"invalid {label} path")
    return path


def _check_wheel(path: Path, *, name: str, version: str) -> None:
    with zipfile.ZipFile(path) as wheel:
        members = set(wheel.namelist())
        _require_members(
            members,
            {
                f"{name}/__init__.py",
                f"{name}/cli.py",
                f"{name}/py.typed",
            },
            label="wheel",
        )

        metadata_member = _one_matching(
            members,
            suffix=".dist-info/METADATA",
            label="wheel METADATA",
        )
        metadata = BytesParser().parsebytes(wheel.read(metadata_member))
        if metadata.get("Name") != name:
            raise DistributionCheckError("wheel metadata name does not match pyproject")
        if metadata.get("Version") != version:
            raise DistributionCheckError("wheel metadata version does not match pyproject")

        entry_points_member = _one_matching(
            members,
            suffix=".dist-info/entry_points.txt",
            label="wheel entry_points.txt",
        )
        entry_points = wheel.read(entry_points_member).decode("utf-8")
        if "nerlex = nerlex.cli:main" not in entry_points:
            raise DistributionCheckError("wheel is missing the nerlex CLI entry point")


def _check_sdist(path: Path, *, name: str, version: str) -> None:
    prefix = f"{name}-{version}/"
    with tarfile.open(path, mode="r:gz") as archive:
        members = {member.name for member in archive.getmembers()}

    _require_members(
        members,
        {
            f"{prefix}LICENSE",
            f"{prefix}README.md",
            f"{prefix}pyproject.toml",
            f"{prefix}src/{name}/__init__.py",
            f"{prefix}src/{name}/cli.py",
            f"{prefix}src/{name}/py.typed",
        },
        label="source distribution",
    )


def _one_matching(members: set[str], *, suffix: str, label: str) -> str:
    matches = tuple(member for member in members if member.endswith(suffix))
    if len(matches) != 1:
        raise DistributionCheckError(
            f"expected exactly one {label}, found {len(matches)}"
        )
    return matches[0]


def _require_members(members: set[str], required: set[str], *, label: str) -> None:
    missing = sorted(required - members)
    if missing:
        raise DistributionCheckError(f"{label} is missing required files: {missing}")


if __name__ == "__main__":
    main()

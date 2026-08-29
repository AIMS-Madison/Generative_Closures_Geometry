"""Repository-relative path helpers.

The environment overrides are useful when full training data or checkpoints
live outside a lightweight clone of the repository.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def get_project_root() -> Path:
    """Return the repository root, optionally overridden by GCG_PROJECT_ROOT."""
    override = os.environ.get("GCG_PROJECT_ROOT")
    return Path(override).expanduser().resolve() if override else PROJECT_ROOT


def project_path(*parts: str | os.PathLike[str]) -> Path:
    """Build an absolute path inside the repository."""
    return get_project_root().joinpath(*parts)


def data_path(*parts: str | os.PathLike[str]) -> Path:
    """Build a path below the data root."""
    override = os.environ.get("GCG_DATA_ROOT")
    root = Path(override).expanduser().resolve() if override else project_path("Data")
    return root.joinpath(*parts)


def model_path(*parts: str | os.PathLike[str]) -> Path:
    """Build a path below the trained-model root."""
    override = os.environ.get("GCG_MODEL_ROOT")
    root = Path(override).expanduser().resolve() if override else project_path("Trained_Models")
    return root.joinpath(*parts)


def resolve_input_path(
    env_var: str,
    default_relative_path: str | os.PathLike[str],
) -> Path:
    """Resolve an input path from an environment variable or repository default."""
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser().resolve()
    path = Path(default_relative_path)
    return path if path.is_absolute() else project_path(path)


def resolve_output_path(
    relative_path: str | os.PathLike[str],
    env_var: str = "GCG_OUTPUT_ROOT",
) -> Path:
    """Resolve an output path and create its parent directory."""
    path = Path(relative_path)
    override = os.environ.get(env_var)
    if override and not path.is_absolute():
        path = Path(override).expanduser().resolve() / path
    elif not path.is_absolute():
        path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

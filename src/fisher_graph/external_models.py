"""Safety helpers for opt-in external model downloads.

The project intentionally keeps third-party model payloads outside its Git
worktree.  These helpers resolve the Hugging Face cache using the same
environment variables users commonly configure, then fail closed if that
location is the repository or one of its descendants.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Mapping
from pathlib import Path


def _huggingface_home(
    *,
    environment: Mapping[str, str],
    home: Path | None,
) -> Path:
    configured = environment.get("HF_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    xdg_cache = environment.get("XDG_CACHE_HOME")
    if xdg_cache:
        return (
            Path(xdg_cache).expanduser() / "huggingface"
        ).resolve()
    resolved_home = (
        Path.home() if home is None else Path(home).expanduser()
    ).resolve()
    return resolved_home / ".cache" / "huggingface"


def find_git_worktree(start: Path | None = None) -> Path | None:
    """Return the nearest parent containing ``.git``, if one exists."""

    current = (Path.cwd() if start is None else Path(start)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def default_huggingface_cache_dir(
    *,
    environment: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Resolve the effective Hugging Face Hub cache without creating it."""

    values = os.environ if environment is None else environment
    for variable in (
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ):
        configured = values.get(variable)
        if configured:
            return Path(configured).expanduser().resolve()

    return _huggingface_home(
        environment=values,
        home=home,
    ) / "hub"


def huggingface_local_paths(
    cache_dir: Path | str | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> dict[str, Path]:
    """Resolve every local path Hugging Face may write during a download."""

    values = os.environ if environment is None else environment
    hf_home = _huggingface_home(environment=values, home=home)
    hub_cache = (
        default_huggingface_cache_dir(
            environment=values,
            home=home,
        )
        if cache_dir is None
        else Path(cache_dir).expanduser().resolve()
    )
    return {
        "hub_cache": hub_cache,
        "assets_cache": Path(
            values.get("HF_ASSETS_CACHE", str(hf_home / "assets"))
        ).expanduser().resolve(),
        "xet_cache": Path(
            values.get("HF_XET_CACHE", str(hf_home / "xet"))
        ).expanduser().resolve(),
        "token_path": Path(
            values.get("HF_TOKEN_PATH", str(hf_home / "token"))
        ).expanduser().resolve(),
    }


def external_huggingface_cache_dir(
    cache_dir: Path | str | None = None,
    *,
    repository_root: Path | None = None,
    additional_repository_roots: Collection[Path] = (),
    environment: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Resolve a cache and reject any location inside the active repository.

    Passing the returned path explicitly to ``from_pretrained`` avoids a
    process-level Hugging Face setting silently putting hundreds of megabytes
    of weights below the checkout.
    """

    local_paths = huggingface_local_paths(
        cache_dir,
        environment=environment,
        home=home,
    )
    resolved = local_paths["hub_cache"]
    primary_root = (
        find_git_worktree()
        if repository_root is None
        else Path(repository_root).resolve()
    )
    roots = []
    if primary_root is not None:
        roots.append(primary_root)
    for candidate in additional_repository_roots:
        root = Path(candidate).resolve()
        if root not in roots:
            roots.append(root)
    for root in roots:
        unsafe = {
            name: path
            for name, path in local_paths.items()
            if path == root or root in path.parents
        }
        if unsafe:
            details = ", ".join(
                f"{name}={path}" for name, path in unsafe.items()
            )
            raise ValueError(
                "Hugging Face model cache and credentials must be outside "
                f"the Git worktree: {details}, worktree={root}"
            )
    return resolved


__all__ = [
    "default_huggingface_cache_dir",
    "external_huggingface_cache_dir",
    "find_git_worktree",
    "huggingface_local_paths",
]

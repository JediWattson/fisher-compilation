import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fisher_graph.external_models import (
    default_huggingface_cache_dir,
    external_huggingface_cache_dir,
    find_git_worktree,
    huggingface_local_paths,
)


class ExternalModelCacheTests(unittest.TestCase):
    def test_default_cache_honors_hub_then_home_then_xdg(self) -> None:
        home = Path("/users/example")
        self.assertEqual(
            default_huggingface_cache_dir(
                environment={"HF_HUB_CACHE": "/external/hub"},
                home=home,
            ),
            Path("/external/hub"),
        )
        self.assertEqual(
            default_huggingface_cache_dir(
                environment={"HF_HOME": "/external/hf"},
                home=home,
            ),
            Path("/external/hf/hub"),
        )
        self.assertEqual(
            default_huggingface_cache_dir(
                environment={"XDG_CACHE_HOME": "/external/cache"},
                home=home,
            ),
            Path("/external/cache/huggingface/hub"),
        )
        self.assertEqual(
            default_huggingface_cache_dir(environment={}, home=home),
            Path("/users/example/.cache/huggingface/hub"),
        )

    def test_explicit_cache_inside_repository_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "outside the Git"):
                external_huggingface_cache_dir(
                    root / "models",
                    repository_root=root,
                )
            with self.assertRaisesRegex(ValueError, "outside the Git"):
                external_huggingface_cache_dir(
                    root,
                    repository_root=root,
                )

    def test_environment_cache_inside_repository_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "outside the Git"):
                external_huggingface_cache_dir(
                    repository_root=root,
                    environment={
                        "HF_HOME": str(root / ".huggingface"),
                    },
                )

    def test_all_huggingface_write_paths_are_guarded(self) -> None:
        variables = (
            ("HF_ASSETS_CACHE", "assets_cache"),
            ("HF_XET_CACHE", "xet_cache"),
            ("HF_TOKEN_PATH", "token_path"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            external = root.parent / f"{root.name}-external-hub"
            for variable, label in variables:
                with self.subTest(variable=variable):
                    with self.assertRaisesRegex(ValueError, label):
                        external_huggingface_cache_dir(
                            external,
                            repository_root=root,
                            environment={
                                variable: str(root / variable.lower()),
                                "HF_HOME": str(
                                    root.parent / f"{root.name}-external-home"
                                ),
                            },
                        )

    def test_local_path_report_uses_explicit_external_cache(self) -> None:
        paths = huggingface_local_paths(
            "/external/models",
            environment={"HF_HOME": "/external/hf"},
        )

        self.assertEqual(paths["hub_cache"], Path("/external/models"))
        self.assertEqual(paths["assets_cache"], Path("/external/hf/assets"))
        self.assertEqual(paths["xet_cache"], Path("/external/hf/xet"))
        self.assertEqual(paths["token_path"], Path("/external/hf/token"))

    def test_external_cache_is_returned_without_being_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            repository = parent / "repository"
            repository.mkdir()
            cache = parent / "external-cache"

            resolved = external_huggingface_cache_dir(
                cache,
                repository_root=repository,
            )

            self.assertEqual(resolved, cache.resolve())
            self.assertFalse(cache.exists())

    def test_additional_package_worktree_is_also_guarded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            current_repository = parent / "current"
            package_repository = parent / "package"
            current_repository.mkdir()
            package_repository.mkdir()

            with self.assertRaisesRegex(ValueError, str(package_repository)):
                external_huggingface_cache_dir(
                    package_repository / "models",
                    repository_root=current_repository,
                    additional_repository_roots=(package_repository,),
                )

    def test_find_git_worktree_walks_upward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            child = root / "a" / "b"
            child.mkdir(parents=True)

            self.assertEqual(find_git_worktree(child), root.resolve())

    def test_autodetection_uses_current_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            child = root / "nested"
            child.mkdir()
            with patch.dict(os.environ, {}, clear=True), patch(
                "pathlib.Path.cwd",
                return_value=child,
            ):
                with self.assertRaisesRegex(ValueError, "outside the Git"):
                    external_huggingface_cache_dir(root / "weights")


if __name__ == "__main__":
    unittest.main()

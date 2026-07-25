import fnmatch
import subprocess
import unittest
from pathlib import Path, PurePosixPath


class RepositoryModelHygieneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        if not (cls.root / ".git").exists():
            raise unittest.SkipTest("repository metadata is not available")

    def test_no_external_model_payload_is_tracked(self) -> None:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=self.root,
            check=True,
            capture_output=True,
        )
        tracked = tuple(
            name
            for name in completed.stdout.decode("utf-8").split("\0")
            if name
        )
        forbidden_directories = {
            ".cache",
            ".hf-cache",
            ".huggingface",
            ".xet-cache",
        }
        forbidden_roots = {
            ".local-runs",
            "local-runs",
            "models",
            "checkpoints",
        }
        forbidden_globs = (
            "*.safetensors",
            "*.safetensors.index.json",
            "pytorch_model*.bin",
            "*.gguf",
            "tokenizer.model",
        )
        violations = []
        for name in tracked:
            path = PurePosixPath(name)
            if (
                any(part in forbidden_directories for part in path.parts)
                or (path.parts and path.parts[0] in forbidden_roots)
            ):
                violations.append(name)
                continue
            if any(
                fnmatch.fnmatch(path.name, pattern)
                for pattern in forbidden_globs
            ):
                violations.append(name)

        self.assertEqual(
            violations,
            [],
            "external model payloads must not be tracked",
        )

    def test_common_model_and_local_run_paths_are_ignored(self) -> None:
        candidates = (
            ".hf-cache/models--google--gemma-3-270m/blobs/weights",
            ".huggingface/token",
            ".xet-cache/chunk-cache/chunk",
            "models/gemma-3-270m/model.safetensors",
            "checkpoints/pytorch_model-00001-of-00002.bin",
            ".local-runs/gemma-3-270m/layer-0-fisher.pt",
            "root-copy.gguf",
            "tokenizer.model",
        )
        completed = subprocess.run(
            ["git", "check-ignore", "--stdin"],
            cwd=self.root,
            check=False,
            input="\n".join(candidates) + "\n",
            capture_output=True,
            text=True,
        )
        ignored = set(completed.stdout.splitlines())

        self.assertEqual(ignored, set(candidates))


if __name__ == "__main__":
    unittest.main()

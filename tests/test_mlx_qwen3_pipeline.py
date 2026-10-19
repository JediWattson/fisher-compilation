"""Local CLI validation, including a tiny real Qwen conversion/fit/decode."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from fisher_graph.mlx_qwen3_pipeline import _layers, _read_corpus, main


def test_help_does_not_load_mlx():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; from fisher_graph.mlx_qwen3_pipeline import build_parser; "
         "build_parser().format_help(); assert 'mlx.core' not in sys.modules"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_layer_selection():
    assert _layers("0,2,4-6,2", 8) == (0, 2, 4, 5, 6)
    assert _layers("all", 3) == (0, 1, 2)
    for value in ("", "2-0", "-1", "0,8", "1-2-3"):
        with pytest.raises(ValueError):
            _layers(value, 8)


def test_corpus_bounds_and_explicit_positions(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({
        "input_ids": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        "attention_mask": [[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        "position_ids": [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]],
    }) + "\n")
    batches = _read_corpus(path, max_examples=2, max_length=4, max_tokens=7)
    assert batches == [
        {"input_ids": [[0, 1, 2, 3]], "attention_mask": [[0, 1, 1, 1]], "position_ids": [[0, 2, 4, 6]]},
        {"input_ids": [[5, 6, 7]], "attention_mask": [[1, 1, 1]], "position_ids": [[1, 3, 5]]},
    ]
    assert len(_read_corpus(path, max_examples=1)) == 1
    with pytest.raises(ValueError, match="at least two"):
        _read_corpus(path, max_tokens=1)
    with pytest.raises(ValueError, match="outside.*vocabulary"):
        _read_corpus(path, vocab_size=3)


@pytest.mark.parametrize("item,match", [
    ({"text": "example"}, "requires --tokenizer"),
    ({"input_ids": [1, -1]}, "nonnegative integers"),
    ({"input_ids": [1, 2], "attention_mask": [1]}, "align"),
    ({"input_ids": [1, 2], "attention_mask": [1, 2]}, "binary"),
    ({"input_ids": [1, 2], "position_ids": [0, -1]}, "nonnegative integers"),
    ({"input_ids": [1, 2], "attention_mask": [0, 1]}, "at least two"),
])
def test_invalid_corpus(tmp_path, item, match):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(item) + "\n")
    with pytest.raises(ValueError, match=match):
        _read_corpus(path)


def test_real_local_convert_fit_and_generate(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    from fisher_graph.mlx_executor import mlx_metal_is_usable
    if not mlx_metal_is_usable():
        pytest.skip("MLX/Metal is unavailable")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from fisher_graph.mlx_qwen3 import MLXQwen3

    source = tmp_path / "hf"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(731)
        config = transformers.Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=4, max_position_embeddings=64,
            tie_word_embeddings=True, attention_dropout=0.0,
            pad_token_id=0, eos_token_id=2,
        )
        model = transformers.Qwen3ForCausalLM(config).eval()
    model.save_pretrained(source, safe_serialization=True)
    del model
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, **{f"word{i}": i for i in range(3, 32)}}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]",
    )
    tokenizer.save_pretrained(source)
    original = tmp_path / "exact"
    fitted = tmp_path / "fitted"
    assert main(["convert", "--hf-snapshot", str(source), "--tokenizer", str(source),
                 "--output", str(original)]) == 0
    corpus = tmp_path / "calibration.jsonl"
    corpus.write_text("\n".join(json.dumps({"input_ids": list(range(i, i + 8))}) for i in range(3, 7)) + "\n")
    assert main([
        "fit", "--artifact", str(original), "--corpus", str(corpus), "--layers", "0",
        "--attention-input-rank", "8", "--attention-output-rank", "6", "--chart-count", "2",
        "--chart-rank", "4", "--generator-rank", "4", "--trajectory-steps", "2",
        "--output", str(fitted),
    ]) == 0
    report = json.loads((fitted / "pipeline_report.json").read_text())
    assert report["calibration_examples"] == 4
    assert report["calibration_physical_tokens"] == 32
    assert report["mlx_enable_tf32"] == "0"
    assert report["heldout_confirmation"] is False
    assert len(report["fits"]) == 2
    assert report["trajectory"] is not None
    runtime = MLXQwen3.load(fitted)
    assert runtime.parameters["attention.0.input_basis"].shape == (16, 8)
    assert "mlp.0.gate_weight" not in runtime.parameters
    assert "mlp.1.gate_weight" in runtime.parameters
    assert "attention.0.input_basis" not in MLXQwen3.load(original).parameters
    completion = tmp_path / "results" / "completion.json"
    capsys.readouterr()
    assert main([
        "generate", "--artifact", str(fitted), "--tokenizer", str(source),
        "--prompt", "word3 word4", "--max-new-tokens", "3", "--ignore-eos", "--report", str(completion),
    ]) == 0
    generated = json.loads(completion.read_text())
    assert generated["prompt_tokens"] == 2
    assert len(generated["generated_token_ids"]) == 3
    assert generated["decoding"] == "greedy"
    assert generated["completion"] == capsys.readouterr().out.strip()
    xray_report = tmp_path / "results" / "xray.json"
    assert main([
        "xray", "--artifact", str(fitted), "--tokenizer", str(source),
        "--prompt", "word3 word4", "--report", str(xray_report),
    ]) == 0
    xray_data = json.loads(xray_report.read_text())
    assert xray_data["operation"] == "xray"
    assert xray_data["reduced_attention_layers"] == [0]
    assert xray_data["chart_mlp_layers"] == [0]
    assert len(xray_data["top5_predictions"]) == 5
    assert len(xray_data["ablation_impacts"]) == 2
    assert os.environ["MLX_ENABLE_TF32"] == "0"

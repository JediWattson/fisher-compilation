from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fisher_graph.adapters import module_state_fingerprint
from fisher_graph.gemma3_global_cross_block_merge_executor import (
    Gemma3GlobalMergedMLP,
)


class _GemmaMLP(nn.Module):
    def __init__(self, width: int = 4, intermediate: int = 6) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.up_proj = nn.Linear(width, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)

    def features(self, values: Tensor) -> Tensor:
        return F.gelu(
            self.gate_proj(values),
            approximate="tanh",
        ) * self.up_proj(values)

    def forward(self, values: Tensor) -> Tensor:
        return self.down_proj(self.features(values))


def _storage_pointers(module: nn.Module) -> set[int]:
    return {
        value.untyped_storage().data_ptr()
        for value in module.state_dict().values()
        if value.numel()
    }


def test_compiled_mlp_physically_removes_many_generator_rows() -> None:
    torch.manual_seed(197)
    source = _GemmaMLP().eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    source_fingerprint = module_state_fingerprint(source)
    compiled = Gemma3GlobalMergedMLP(
        source,
        consumer_source_indices=(1, 4),
        activation="gelu_pytorch_tanh",
    )
    values = torch.randn(2, 3, 4)
    retained = compiled.retained_features(values)
    full = compiled.native_width_input(retained)
    replacement_one = torch.randn(2, 3)
    replacement_four = torch.randn(2, 3)
    full[..., 1] = replacement_one
    full[..., 4] = replacement_four
    expected_features = source.features(values)
    expected_features[..., 1] = replacement_one
    expected_features[..., 4] = replacement_four

    torch.testing.assert_close(
        compiled.down_proj(full),
        source.down_proj(expected_features),
    )
    assert tuple(compiled.retained_source_indices.tolist()) == (0, 2, 3, 5)
    assert compiled.gate_proj.weight.shape == (4, 4)
    assert compiled.up_proj.weight.shape == (4, 4)
    assert compiled.down_proj.weight.shape == (4, 6)
    assert sum(parameter.numel() for parameter in source.parameters()) == 72
    assert sum(parameter.numel() for parameter in compiled.parameters()) == 56
    assert module_state_fingerprint(source) == source_fingerprint
    assert not (_storage_pointers(source) & _storage_pointers(compiled))

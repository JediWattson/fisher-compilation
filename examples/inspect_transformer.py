"""Run a forward/backward trace and activation-Fisher estimate."""

import torch
import torch.nn.functional as F

from fisher_graph import (
    GraphLayerExecutor,
    ToyTransformer,
    TransformerConfig,
    empirical_activation_fisher,
)


torch.manual_seed(7)
model = ToyTransformer(
    TransformerConfig(
        vocab_size=32,
        max_sequence_length=8,
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
    )
)

tokens = torch.randint(0, 32, (3, 8))
inputs, targets = tokens[:, :-1], tokens[:, 1:]

output = model(inputs, capture_activations=True)
assert output.activations is not None
loss = F.cross_entropy(
    output.logits.flatten(0, 1),
    targets.flatten(),
)
loss.backward()

print("Activation taps:")
for name, tensor in output.activations.items():
    gradient = tensor.grad
    gradient_norm = gradient.norm().item() if gradient is not None else float("nan")
    print(f"  {name:36} shape={str(tuple(tensor.shape)):18} |grad|={gradient_norm:.4g}")

report = empirical_activation_fisher(
    model,
    inputs,
    targets,
    activations=lambda name: name.endswith(("output", "probabilities")),
)
print("\nMean empirical diagonal Fisher:")
for name, mean_fisher in report.ranked():
    print(f"  {name:36} {mean_fisher:.6g}")

# Replace layer 0 with an exactly equivalent DAG executor.
block = model.layers[0]
graph = GraphLayerExecutor.from_transformer_block(block)
model.replace_layer(0, graph)
graph_output = model(inputs)
print(f"\nGraph-swapped logits shape: {tuple(graph_output.logits.shape)}")


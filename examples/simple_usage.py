"""
Minimal runnable example showing how the decorators behave.
"""

import torch

from vistvis import visTvis_layer_counter, visTvis_store


@visTvis_store(base_folder_path="./vistvis_runs", identifier="demo", pos_ret=1, var_name="attn")
def attention_block(query: torch.Tensor, attn_weight: torch.Tensor):
    # Pretend this is doing self-attention.
    return torch.matmul(attn_weight, query), attn_weight


@visTvis_layer_counter(identifier="demo", var_name="layer_idx")
def run_model(layers: int, layer_idx: int = 0):
    for _ in range(layers):
        query = torch.rand(2, 4, 4)
        attn_weight = torch.rand(2, 4, 4)
        attention_block(query, attn_weight)
        layer_idx += 1
    return int(layer_idx)


if __name__ == "__main__":
    final_idx = run_model(layers=2, layer_idx=0)
    print(f"Finished at layer index {final_idx}")

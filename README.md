# visTvis

Store (and soon visualize) attention weights from Vision Transformers in a model-agnostic way. The package ships two decorators:

- `@visTvis_store` saves a return value to disk with a predictable folder layout.
- `@visTvis_layer_counter` injects an int-like counter that keeps track of which layer is currently active so saved files end up in the right slot.

## Installation

```bash
pip install vistvis
```

For local development in this repo:

```bash
make install
```

## Quick start

Decorate the attention routine you want to log and point it to a base directory:

```python
from vistvis import visTvis_store


@visTvis_store(base_folder_path="./runs", identifier="global", pos_ret=1, var_name="attn_weight")
def attn_func(self, query, key, value, attn_mask=None, dropout_p=0.0,
              is_causal=False, scale=None, enable_gqa=False):
    # ... compute attn_weight ...
    return attn_weight @ value, attn_weight
```

Next, wrap the function that calls your attention blocks so it receives a `LayerCounter`. The counter behaves like an int (supports indexing, comparisons, `+= 1`, etc.) and updates the visTvis layer state on every mutation:

```python
from vistvis import visTvis_layer_counter


@visTvis_layer_counter(identifier="global", var_name="global_idx")
def run_blocks(self, tokens, pos, resolution, kernel_size, is_causal, attend_to_special_tokens, global_idx=0):
    B, S, P, C = tokens.shape
    intermediates = []

    for _ in range(self.aa_block_size):
        tokens = self.global_blocks[global_idx](tokens, pos=pos, num_frames=S, resolution=resolution,
                                                kernel_size=kernel_size,
                                                attend_to_special_tokens=attend_to_special_tokens,
                                                is_causal=is_causal)
        global_idx += 1  # updates the visTvis layer state
        intermediates.append(tokens.view(B, S, P, C))

    return tokens, int(global_idx), intermediates
```

During runtime each call to `attn_func` will save `attn_weight` under:

```
<base_folder_path>/<identifier>/layer_<layer_idx>/<var_name>.pt
```

By default the layer counter starts at zero and persists between calls for the same identifier. If you need manual control you can use `from vistvis import state` and call `state.update(identifier, value)` before invoking decorated functions.

## Publishing

- Update `pyproject.toml` metadata.
- Build and upload with `python -m build` followed by `twine upload dist/*`.

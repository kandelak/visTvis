# visTvis (Visual Transformer Visualizer)

Store and visualize attention weights from Vision Transformers in a model-agnostic way. The package ships two decorators:

- `@visTvis_store` saves a return value tensor to disk with a predictable folder layout.
- `@visTvis_layer_counter` injects an int-like counter that keeps track of which layer is currently active so saved files end up in the right slot.
- Visualization helpers overlay attention on input frames and stack results into PDFs.

## Demo

### Choose Query Patches Interactively from frame N
![Interactive example](https://raw.githubusercontent.com/kandelak/visTvis/refs/heads/main/content/vis_demo.gif)

### Get per-layer-head attention overlays on frame M
![Overlay example](https://raw.githubusercontent.com/kandelak/visTvis/refs/heads/main/content/img.png)


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


@visTvis_store(base_folder_path="./runs", identifier="demo", file_name="attn.pt", folder_name="attn")
def attn_func(attn_weight):
    # ... compute attn_weight ...
    return attn_weight
```

Next, wrap the function that calls your attention blocks so it receives a `LayerCounter`. The counter behaves like an int (supports indexing, comparisons, `+= 1`, etc.) and updates the visTvis layer state on every mutation:

```python
from vistvis import visTvis_layer_counter


@visTvis_layer_counter(identifier="demo", var_name="layer_idx")
def run_blocks(blocks, inputs, layer_idx=0):
    for block in blocks:
        attn_weight = block(inputs)
        attn_func(attn_weight)  # stored under runs/demo/attn/layer_<idx>/attn.pt
        layer_idx += 1  # updates the visTvis layer state
    return int(layer_idx)
```

During runtime each call to `attn_func` will save `attn_weight` under:

```
<base_folder_path>/<identifier>/<folder_name>/layer_<layer_idx>/<file_name>
```

By default the layer counter starts at zero and persists between calls for the same identifier. If you need manual control you can use `from vistvis import state` and call `state.update(identifier, value)` before invoking decorated functions.

## Visualize saved attention

Save attention under `attn/` and inputs under `input/` so the helpers can discover them:

```python
from vistvis import AttentionPath, run_overlay_for_reconstruction, plot_attention_for_reconstruction, StaticMetadata

paths = AttentionPath(base_dir="./runs", experiment_name="demo")
metadata = StaticMetadata(patch_size=16, num_special_tokens=0)  # or set env VIS_TVIS_METADATA_PATH to a YAML file

run_overlay_for_reconstruction(
    paths,
    query_frame_idx=0,
    query_idx=0,  # or "interactive" to select via GUI
    key_frame_idx=0,
    head="mean",  # or int / "all"
    output_path="./overlays_demo",
    metadata=metadata,
)

plot_attention_for_reconstruction("./overlays_demo")
```

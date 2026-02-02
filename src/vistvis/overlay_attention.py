from __future__ import annotations

import os
from typing import Iterable, Literal, Sequence, Union

import numpy as np
from PIL import Image
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import torch

from .static_metadata import StaticMetadata

HeadSpec = Union[int, Literal["mean"]]
QueryIdxSpec = Union[int, list[int], Literal["interactive"]]


def overlay_query_to_frame_attention(
        attn_path: str,
        input_path: str,
        metadata: StaticMetadata | None,
        query_frame_idx: int,
        query_idx: QueryIdxSpec,
        key_frame_idx: int,
        head: HeadSpec,
        output_path: str,
        *,
        batch_idx: int = 0,
        alpha: float = 0.45,
        cmap_name: str = "jet",
        eps: float = 1e-8,
        is_cross_attention: bool = False,
        query_output_path: str | None = None,
) -> list[str]:
    """Overlay one or more query tokens' attention over a target frame image.

    This function loads:
      1) Attention weights with shape (Batch, Heads, P_q, P_k)
        2) Input frames with shape (Batch, N_fr, Channel, Height, Width)

    Then it visualizes the attention from one or more query tokens
    (query_frame_idx, query_idx) to the patch tokens of a specific key frame
    (key_frame_idx), discarding special tokens that are prepended per frame, and
    upsampling the coarse patch-grid attention to (Height, Width) based on patch_size.
    Set is_cross_attention=True to handle cross-attention where queries come from a
    single frame and keys come from a (possibly different) frame with shapes
    (B, Heads, tokens_per_frame_q, tokens_per_frame_k).

    Important indexing details:
      - Each frame contributes tokens_per_frame = num_special_tokens + num_patches tokens.
      - Tokens are ordered per-frame, with special tokens first, followed by patch tokens.
      - query_idx is the *within-frame* token index in [0, tokens_per_frame-1].
      - If you want to specify the k-th patch token within a frame (0..num_patches-1),
        pass query_idx = num_special_tokens + k.

    Args:
        attn_path: Path to attention weights (e.g., .npy/.npz/.pt/.pth).
        input_path: Path to input frames (e.g., .npy/.npz/.pt/.pth).
        metadata: StaticMetadata pydantic model instance with:
            - patch_size (int): patch size of ViT tokens
            - num_special_tokens (int): number of special tokens prepended per frame
            - If None, will attempt to read from VIS_TVIS_METADATA_PATH env var.
        query_frame_idx: Which frame the query token belongs to (0..N_fr-1).
        query_idx: Either a single token index, a list of token indices, or the literal
            "interactive". Interactive mode pops up a matplotlib UI overlaying a patch
            grid for the query frame; clicking multiple patches selects multiple
            query indices (mapped as num_special_tokens + patch_index).
        key_frame_idx: Which frame to visualize keys from (0..N_fr-1).
        head: Either an integer head index or "mean" to average over all heads.
        output_path: Base folder where results are written. Overlays are saved to
            output_path/overlays/overlay_{query_idx}.png. Query frame visualizations
            default to output_path/query_frames/query_image_{query_idx}.png with a
            green rectangle marking the query location (full-frame border for special
            tokens with no spatial location).
        batch_idx: Which batch element to visualize (default: 0).
        alpha: Heatmap opacity blended over the image (default: 0.45).
        cmap_name: Matplotlib colormap name (default: "jet").
        eps: Small constant for numeric stability in normalization (default: 1e-8).
        is_cross_attention: Whether the attention tensor is cross-attention between
            query_frame_idx (queries) and key_frame_idx (keys). If True, expects
            P_total_q == tokens_per_frame == num_special_tokens + num_patches and
            P_total_k == the same tokens_per_frame for the key frame.
        query_output_path: Optional override for where to save query frame images;
            defaults to {output_path}/query_frames.

    Returns:
        List of paths to overlay images, one per query.

    Raises:
        ValueError: If shapes/indices are invalid or sizes are inconsistent with patch_size.
        FileNotFoundError: If attn_path or input_path do not exist.
        ImportError: If torch is required for loading .pt but not installed.
    """
    if not os.path.exists(attn_path):
        raise FileNotFoundError(f"Attention file not found: {attn_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not metadata:
        env_var = "VIS_TVIS_METADATA_PATH"
        metadata_path = os.getenv(env_var)
        if not metadata_path:
            raise ValueError(
                f"metadata not provided and environment variable {env_var} is not set."
            )
        metadata = StaticMetadata.from_yaml_file(metadata_path)

    attn = _load_array(attn_path)  # expected: (B, Heads, P_total, P_total)
    x = _load_array(input_path)  # expected: (B, N_fr, C, H, W)

    if attn.ndim != 4:
        raise ValueError(f"Expected attn to have 4 dims (B, Heads, P, P); got {attn.shape}")
    if x.ndim != 5:
        raise ValueError(f"Expected input to have 5 dims (B, N_fr, C, H, W); got {x.shape}")

    B, Hds, P_total_q, P_total_k = attn.shape
    if not is_cross_attention and P_total_q != P_total_k:
        raise ValueError(f"Expected square attention over tokens; got {attn.shape}")
    if not (0 <= batch_idx < B):
        raise ValueError(f"batch_idx out of range: {batch_idx} (B={B})")

    _, N_fr, _, H, W = x.shape
    if not (0 <= key_frame_idx < N_fr):
        raise ValueError(f"key_frame_idx out of range: {key_frame_idx} (N_fr={N_fr})")
    if not (0 <= query_frame_idx < N_fr):
        raise ValueError(f"query_frame_idx out of range: {query_frame_idx} (N_fr={N_fr})")

    patch_size = int(getattr(metadata, "patch_size"))
    num_special = int(getattr(metadata, "num_special_tokens"))
    if patch_size <= 0:
        raise ValueError(f"metadata.patch_size must be > 0; got {patch_size}")
    if num_special < 0:
        raise ValueError(f"metadata.num_special_tokens must be >= 0; got {num_special}")

    # Compute patch grid. Must be divisible.
    if (H % patch_size) != 0 or (W % patch_size) != 0:
        raise ValueError(
            f"Input frame size (H={H}, W={W}) must be divisible by patch_size={patch_size}."
        )
    grid_h = H // patch_size
    grid_w = W // patch_size
    num_patches = grid_h * grid_w
    tokens_per_frame = num_special + num_patches

    queries: Sequence[int]
    if isinstance(query_idx, str):
        if query_idx != "interactive":
            raise ValueError(f"query_idx string must be 'interactive'; got {query_idx!r}")
        frame_for_selection = x[batch_idx, query_frame_idx]
        queries = _select_queries_interactively(
            frame_for_selection,
            grid_w=grid_w,
            grid_h=grid_h,
            patch_size=patch_size,
            num_special=num_special,
        )
        if not queries:
            raise ValueError("No queries selected in interactive mode.")
    elif isinstance(query_idx, Iterable) and not isinstance(query_idx, (int, np.integer)):
        queries = list(query_idx)
        if not queries:
            raise ValueError("query_idx list is empty.")
    else:
        queries = [int(query_idx)]

    for q in queries:
        if not (0 <= q < tokens_per_frame):
            raise ValueError(
                f"query_idx out of range for a single frame: {q} "
                f"(expected 0..{tokens_per_frame - 1}; tokens_per_frame={tokens_per_frame})"
            )

    if is_cross_attention:
        expected_q = tokens_per_frame
        expected_k = tokens_per_frame
        if P_total_q != expected_q or P_total_k != expected_k:
            raise ValueError(
                "Cross-attention token count mismatch.\n"
                f"  From input+metadata: tokens_per_frame={tokens_per_frame} "
                f"(num_special={num_special}, num_patches={num_patches})\n"
                f"  From attention: P_q={P_total_q}, P_k={P_total_k}\n"
                "For cross-attention, queries and keys should each cover exactly one frame."
            )
    else:
        expected_total = N_fr * tokens_per_frame
        if expected_total != P_total_q:
            raise ValueError(
                "Token count mismatch.\n"
                f"  From input+metadata: N_fr={N_fr}, tokens_per_frame={tokens_per_frame} "
                f"(num_special={num_special}, num_patches={num_patches}) => expected P_total={expected_total}\n"
                f"  From attention: P_total={P_total_q}\n"
                "Check that input H/W, patch_size, num_special_tokens, and N_fr match the attention tensor."
            )

    overlay_dir = os.path.join(output_path, "overlays")
    query_dir = query_output_path or os.path.join(output_path, "query_frames")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)

    saved_paths: list[str] = []
    key_frame = x[batch_idx, key_frame_idx]  # (C, H, W)
    key_img_rgb = _to_uint8_rgb(key_frame)
    query_img_rgb = _to_uint8_rgb(x[batch_idx, query_frame_idx])

    for q in queries:
        query_global_idx = q if is_cross_attention else (query_frame_idx * tokens_per_frame + q)
        if not (0 <= query_global_idx < P_total_q):
            raise ValueError(
                f"Computed query_global_idx out of range: {query_global_idx} (P_total={P_total_q})"
            )

        if head == "mean":
            attn_row = attn[batch_idx].mean(axis=0)[query_global_idx]  # (P_total_k,)
        else:
            if not isinstance(head, int):
                raise ValueError(f"head must be int or 'mean'; got {head!r}")
            if not (0 <= head < Hds):
                raise ValueError(f"head index out of range: {head} (Heads={Hds})")
            attn_row = attn[batch_idx, head, query_global_idx]  # (P_total_k,)
        # attn_row has to sum to 1.0 over P_total_k
        if not np.isclose(float(attn_row.sum()), 1.0, atol=1e-4):
            raise ValueError(
                f"Attention row does not sum to 1.0 (sum={float(attn_row.sum())}) "
                f"for query_global_idx={query_global_idx}, head={head}"
            )

        if is_cross_attention:
            k0 = num_special
            k1 = k0 + num_patches
        else:
            frame_start = key_frame_idx * tokens_per_frame
            k0 = frame_start + num_special
            k1 = k0 + num_patches
        attn_frame_patches = attn_row[k0:k1]  # (num_patches,)

        attn_grid = attn_frame_patches.reshape(grid_h, grid_w)

        attn_min = float(attn_grid.min())
        attn_max = float(attn_grid.max())
        attn_norm = (attn_grid - attn_min) / (attn_max - attn_min + eps)

        attn_up = _resize_attention(attn_norm, (W, H))  # PIL expects (W, H)

        cmap = cm.get_cmap(cmap_name)
        heat_rgba = cmap(attn_up)  # (H, W, 4) floats in [0, 1]
        heat_rgb = (heat_rgba[..., :3] * 255.0).astype(np.uint8)

        out = (
                key_img_rgb.astype(np.float32) * (1.0 - alpha) + heat_rgb.astype(np.float32) * alpha
        )
        out = np.clip(out, 0, 255).astype(np.uint8)

        overlay_file = os.path.join(overlay_dir, f"overlay_{q}.png")
        Image.fromarray(out).save(overlay_file)
        saved_paths.append(overlay_file)

        query_image = _draw_query_highlight(
            query_img_rgb, q, patch_size, num_special, grid_w, grid_h
        )
        query_file = os.path.join(query_dir, f"query_image_{q}.png")
        query_image.save(query_file)

    return saved_paths


def _load_array(path: str) -> np.ndarray:
    """Load a tensor/array from .npy/.npz or torch .pt/.pth into a NumPy array."""
    ext = os.path.splitext(path)[1].lower()

    if ext in {".npy"}:
        arr = np.load(path, allow_pickle=False)
        return np.asarray(arr)

    if ext in {".npz"}:
        z = np.load(path, allow_pickle=False)
        # Prefer a standard key if present; otherwise take the first array.
        for key in ("arr_0", "attn", "attention", "input", "x", "data"):
            if key in z:
                return np.asarray(z[key])
        first_key = list(z.keys())[0]
        return np.asarray(z[first_key])

    if ext in {".pt", ".pth"}:
        if torch is None:
            raise ImportError("torch is required to load .pt/.pth files. Install with `pip install torch`.")
        obj = torch.load(path, map_location="cpu")
        # Common patterns: direct tensor, dict with tensor, etc.
        if hasattr(obj, "detach"):
            return obj.detach().cpu().numpy()
        if isinstance(obj, dict):
            for key in ("attn", "attention", "x", "input", "data", "arr"):
                if key in obj and hasattr(obj[key], "detach"):
                    return obj[key].detach().cpu().numpy()
            for v in obj.values():
                if hasattr(v, "detach"):
                    return v.detach().cpu().numpy()
                if isinstance(v, np.ndarray):
                    return v
        if isinstance(obj, np.ndarray):
            return obj
        raise ValueError(f"Unsupported .pt/.pth content type: {type(obj)}")

    raise ValueError(f"Unsupported file extension {ext!r} for path: {path}")


def _resize_attention(attn_2d: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Resize a 2D attention map to (W, H) using bilinear interpolation."""
    if attn_2d.ndim != 2:
        raise ValueError(f"Expected 2D attention map; got shape {attn_2d.shape}")
    # Convert to 8-bit for PIL resizing, then back to float in [0,1].
    attn_u8 = (np.clip(attn_2d, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil = Image.fromarray(attn_u8, mode="L")
    pil = pil.resize(size_wh, resample=Image.BILINEAR)
    out = np.asarray(pil).astype(np.float32) / 255.0
    return out


def _to_uint8_rgb(frame_chw: np.ndarray) -> np.ndarray:
    """Convert a (C,H,W) frame to uint8 RGB (H,W,3).

    Accepts:
      - float in [0,1] or [0,255]
      - uint8 in [0,255]
      - C in {1,3}

    Returns:
        np.ndarray: uint8 RGB image of shape (H, W, 3).
    """
    if frame_chw.ndim != 3:
        raise ValueError(f"Expected frame (C,H,W); got {frame_chw.shape}")
    C, H, W = frame_chw.shape
    if C not in (1, 3):
        raise ValueError(f"Expected C to be 1 or 3; got C={C}")

    frame = frame_chw
    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        mx = float(np.max(frame)) if frame.size else 0.0
        if mx <= 1.0 + 1e-6:
            frame = frame * 255.0
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)

    if C == 1:
        gray = frame[0]
        rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        rgb = np.transpose(frame, (1, 2, 0))  # (H,W,3)
    return rgb


def _draw_query_highlight(
        img_rgb: np.ndarray,
        query_idx: int,
        patch_size: int,
        num_special: int,
        grid_w: int,
        grid_h: int,
) -> Image.Image:
    """Mark the query location with a green rectangle (full border for special tokens)."""
    from PIL import ImageDraw

    img = Image.fromarray(img_rgb.copy())
    draw_ctx = ImageDraw.Draw(img)

    if query_idx < num_special:
        # Special token: mark the full frame border.
        h, w = img_rgb.shape[0], img_rgb.shape[1]
        draw_ctx.rectangle([(0, 0), (w - 1, h - 1)], outline="green", width=3)
    else:
        patch_idx = query_idx - num_special
        row = patch_idx // grid_w
        col = patch_idx % grid_w
        if not (0 <= row < grid_h and 0 <= col < grid_w):
            raise ValueError(
                f"query_idx {query_idx} maps to invalid patch row/col ({row}, {col}) "
                f"for grid {grid_h}x{grid_w}"
            )
        x0 = col * patch_size
        y0 = row * patch_size
        x1 = x0 + patch_size - 1
        y1 = y0 + patch_size - 1
        draw_ctx.rectangle([(x0, y0), (x1, y1)], outline="green", width=3)

    return img


def _select_queries_interactively(
        frame_chw: np.ndarray,
        *,
        grid_w: int,
        grid_h: int,
        patch_size: int,
        num_special: int,
) -> list[int]:
    """Open a matplotlib UI to pick patch queries; returns query_idx values."""
    rgb = _to_uint8_rgb(frame_chw)
    h, w, _ = rgb.shape

    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.set_title("Click patches to select queries; close window when done.")
    # Draw patch grid lines.
    for x in range(0, w + 1, patch_size):
        ax.axvline(x - 0.5, color="white", alpha=0.4, linewidth=0.8)
    for y in range(0, h + 1, patch_size):
        ax.axhline(y - 0.5, color="white", alpha=0.4, linewidth=0.8)

    selections: list[int] = []

    def onclick(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        col = int(event.xdata // patch_size)
        row = int(event.ydata // patch_size)
        if not (0 <= row < grid_h and 0 <= col < grid_w):
            return
        q = num_special + row * grid_w + col
        selections.append(q)
        rect = Rectangle(
            (col * patch_size, row * patch_size),
            patch_size,
            patch_size,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    plt.close(fig)
    return selections


__all__ = ["overlay_query_to_frame_attention"]

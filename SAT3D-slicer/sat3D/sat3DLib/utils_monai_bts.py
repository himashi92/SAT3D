from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from monai.data.meta_tensor import MetaTensor
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
    optional_import,
    pytorch_after,
)

tqdm, _ = optional_import("tqdm", name="tqdm")
_nearest_mode = "nearest-exact" if pytorch_after(1, 11) else "nearest"

__all__ = ["sliding_window_inference"]

def map_points_to_local_patch(points, patch_slices):
    if points is None:
        return None, None

    points_np = points.cpu().numpy()  # shape (1, N, 3) => (N, 3)
    if points_np.ndim == 3:
        points_np = points_np[0]

    local_points = []
    indices = []
    for idx, pt in enumerate(points_np):
        if all(patch_slices[d].start <= pt[d] < patch_slices[d].stop for d in range(3)):
            local_pt = [int(pt[d]) - patch_slices[d].start for d in range(3)]
            local_points.append(local_pt)
            indices.append(idx)

    if not local_points:
        return None, None

    return torch.tensor(local_points).unsqueeze(0).to(points.device), indices

def sliding_window_inference(
    inputs: torch.Tensor | MetaTensor,
    roi_size: Sequence[int] | int,
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
    overlap: Sequence[float] | float = 0.25,
    mode: BlendMode | str = BlendMode.CONSTANT,
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: torch.device | str | None = None,
    device: torch.device | str | None = None,
    progress: bool = False,
    roi_weight_map: torch.Tensor | None = None,
    process_fn: Callable | None = None,
    buffer_steps: int | None = None,
    buffer_dim: int = -1,
    with_coord: bool = False,
    points: Sequence[torch.Tensor] | None = None,
    low_res_prev_masks: torch.Tensor | None = None,
    low_res_conf: torch.Tensor | None = None,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:

    # Initialization (same as MONAI)
    batch_size, _, *image_size_ = inputs.shape
    device = device or inputs.device
    sw_device = sw_device or inputs.device
    inputs = convert_data_type(inputs, torch.Tensor, wrap_sequence=True)[0]
    roi_size = fall_back_tuple(roi_size, image_size_)
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(len(image_size_)))

    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode), value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, len(image_size_), ensure_tuple_rep(overlap, len(image_size_)))
    slices = dense_patch_slices(image_size, roi_size, scan_interval)

    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    importance_map_ = compute_importance_map(valid_patch_size, mode=mode, sigma_scale=sigma_scale, device=sw_device, dtype=inputs.dtype)
    if len(importance_map_.shape) == len(image_size_):
        importance_map_ = importance_map_.unsqueeze(0).unsqueeze(0)

    output = torch.zeros((batch_size, 1, *image_size), dtype=inputs.dtype, device=device)
    count_map = torch.zeros_like(output)

    for slice_idx in slices:
        win_slice = [slice(None), slice(None)] + list(slice_idx)
        win_data = inputs[win_slice].to(sw_device)

        # Convert point coordinates
        patch_slices = slice_idx
        if points is not None:
            local_pts, valid_idx = map_points_to_local_patch(points[0], patch_slices)
            if local_pts is not None:
                local_labels = points[1][0][valid_idx].unsqueeze(0)
                curr_points = [local_pts, local_labels]
            else:
                curr_points = None
        else:
            curr_points = None

        image_embedding = predictor.image_encoder(win_data)
        sparse_embeddings, dense_embeddings = predictor.prompt_encoder(points=curr_points, boxes=None, masks=low_res_prev_masks, conf=low_res_conf)
        low_res_masks, _ = predictor.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=predictor.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        seg_prob_out = F.interpolate(low_res_masks, size=roi_size, mode='trilinear', align_corners=False)
        weighted_output = seg_prob_out * importance_map_.to(sw_device)

        output[win_slice] += weighted_output.to(device)
        count_map[win_slice] += importance_map_.to(device)

    output /= count_map

    # Remove padding if added
    if any(pad_size):
        zoom_scale = [output.shape[2 + i] / roi_size[i] for i in range(len(roi_size))]
        final_slicing = []
        for sp in range(len(roi_size)):
            si = len(roi_size) - sp - 1
            start = int(round(pad_size[sp * 2] * zoom_scale[si]))
            end = int(round((pad_size[sp * 2] + image_size_[si]) * zoom_scale[si]))
            final_slicing.insert(0, slice(start, end))
        output = output[(slice(None), slice(None), *final_slicing)]

    return output


def _create_buffered_slices(slices, batch_size, sw_batch_size, buffer_dim, buffer_steps):
    """rearrange slices for buffering"""
    slices_np = np.asarray(slices)
    slices_np = slices_np[np.argsort(slices_np[:, buffer_dim, 0], kind="mergesort")]
    slices = [tuple(slice(c[0], c[1]) for c in i) for i in slices_np]
    slices_np = slices_np[:, buffer_dim]

    _, _, _b_lens = np.unique(slices_np[:, 0], return_counts=True, return_index=True)
    b_ends = np.cumsum(_b_lens).tolist()  # possible buffer flush boundaries
    x = [0, *b_ends][:: min(len(b_ends), int(buffer_steps))]
    if x[-1] < b_ends[-1]:
        x.append(b_ends[-1])
    n_per_batch = len(x) - 1
    windows_range = [
        range(b * x[-1] + x[i], b * x[-1] + x[i + 1], sw_batch_size)
        for b in range(batch_size)
        for i in range(n_per_batch)
    ]
    b_slices = []
    for _s, _r in enumerate(windows_range):
        s_s = slices_np[windows_range[_s - 1].stop % len(slices) if _s > 0 else 0, 0]
        s_e = slices_np[(_r.stop - 1) % len(slices), 1]
        b_slices.append((_r.stop, s_s, s_e))  # buffer index, slice start, slice end
    windows_range = itertools.chain(*windows_range)  # type: ignore
    return slices, n_per_batch, b_slices, windows_range


def _compute_coords(coords, z_scale, out, patch):
    """sliding window batch spatial scaling indexing for multi-resolution outputs."""
    for original_idx, p in zip(coords, patch):
        idx_zm = list(original_idx)  # 4D for 2D image, 5D for 3D image
        if z_scale:
            for axis in range(2, len(idx_zm)):
                idx_zm[axis] = slice(
                    int(original_idx[axis].start * z_scale[axis - 2]), int(original_idx[axis].stop * z_scale[axis - 2])
                )
        out[idx_zm] += p


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: Sequence[float]
) -> tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError(f"len(image_size) {len(image_size)} different from spatial dims {num_spatial_dims}.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError(f"len(roi_size) {len(roi_size)} different from spatial dims {num_spatial_dims}.")

    scan_interval = []
    for i, o in zip(range(num_spatial_dims), overlap):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - o))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def _flatten_struct(seg_out):
    dict_keys = None
    seg_probs: tuple[torch.Tensor, ...]
    if isinstance(seg_out, torch.Tensor):
        seg_probs = (seg_out,)
    elif isinstance(seg_out, Mapping):
        dict_keys = sorted(seg_out.keys())  # track predictor's output keys
        seg_probs = tuple(seg_out[k] for k in dict_keys)
    else:
        seg_probs = ensure_tuple(seg_out)
    return dict_keys, seg_probs


def _pack_struct(seg_out, dict_keys=None):
    if dict_keys is not None:
        return dict(zip(dict_keys, seg_out))
    if isinstance(seg_out, (list, tuple)) and len(seg_out) == 1:
        return seg_out[0]
    return ensure_tuple(seg_out)

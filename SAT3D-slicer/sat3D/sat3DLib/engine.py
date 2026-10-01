"""SAT3D-plus inference engine, ported from SAT3D Studio (app/engine/{roi,inference_engine,
preprocessing,measurements}.py) with no Slicer dependency, so it can run on a worker thread.

All coordinates here are (d, h, w) voxel indices, i.e. axis-for-axis matching
``slicer.util.arrayFromVolume`` (same as SimpleITK's GetArrayFromImage).

=== Prompt coordinate order (read before touching to_model_order) ===
Traced in SAT3D Studio against the plus training data loader: images were built with
``tio.ScalarImage.from_sitk`` (tensor axes W, H, D) and prompt coordinates extracted as
``nonzero(...)[:, [2, 1, 0]]``, i.e. in the *reverse* of the image tensor's axis order.
We feed the (D, H, W) array straight to the image encoder, so prompt coordinates must be
reversed to (w, h, d) before they reach the prompt encoder. ``to_model_order`` is the one
place this happens.

Per-run behaviour:
  - If the segment has no ROI yet, or any spatial prompt now falls outside it (or a
    previously-sent point was removed), a new ``roi_target``^3 crop is centred on all
    spatial prompts and the carried state is dropped.
  - Otherwise the crop and its cached image embedding are reused and only the *new* points
    are fed through, one at a time, each conditioned on the running low-res mask and the
    critic's confidence map (RITM-style refinement, as in training).
  - Box and text prompts are resent unchanged on every forward pass.
"""
import contextlib
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

Coord = Tuple[int, int, int]                 # (d, h, w)
Bounds = Tuple[int, int, int, int, int, int]  # (d0, d1, h0, h1, w0, w1)
PointPrompt = Tuple[Coord, bool]             # (coord, positive)

DEFAULT_PERCENTILE_CLIP = (0.5, 99.5)


# ---------------------------------------------------------------- preprocessing
def preprocess_volume(arr: np.ndarray, percentile_clip=DEFAULT_PERCENTILE_CLIP) -> np.ndarray:
    """(D, H, W) raw intensities -> (D, H, W) float32, robust-clipped and z-normalised."""
    import torch
    import torchio as tio
    arr = arr.astype(np.float32)
    fg = arr > 0
    if np.any(fg):
        lo, hi = np.percentile(arr[fg], percentile_clip)
        arr = np.clip(arr, lo, hi)
    image = tio.ScalarImage(tensor=torch.as_tensor(arr)[None])
    image = tio.ZNormalization(masking_method=lambda x: x > 0)(image)
    return image.data[0].numpy()


# ---------------------------------------------------------------- ROI math
def to_model_order(coord: Sequence[float]) -> Tuple[float, float, float]:
    """(d, h, w) -> (w, h, d). See module docstring for why this exists."""
    d, h, w = coord
    return (w, h, d)


def compute_roi(prompt_coords, box=None, target=128, margin=8) -> Tuple[Bounds, bool]:
    """``target``^3 window centred on the prompts (not clamped to the volume).
    Returns (bounds, tightened) -- tightened if the prompts' extent exceeded ``target``."""
    coords = list(prompt_coords) + (list(box) if box is not None else [])
    if not coords:
        raise ValueError("At least one point, scribble or box prompt is required.")
    arr = np.array(coords, dtype=np.float64)
    lo, hi = arr.min(axis=0) - margin, arr.max(axis=0) + margin
    tightened = bool(np.any(hi - lo > target))
    lo_final = np.round((lo + hi) / 2.0 - target / 2.0).astype(int)
    hi_final = lo_final + target
    (d0, h0, w0), (d1, h1, w1) = lo_final.tolist(), hi_final.tolist()
    return (d0, d1, h0, h1, w0, w1), tightened


def bounds_contains(bounds: Bounds, coord: Coord) -> bool:
    d0, d1, h0, h1, w0, w1 = bounds
    d, h, w = coord
    return d0 <= d < d1 and h0 <= h < h1 and w0 <= w < w1


def to_local(coord: Coord, bounds: Bounds) -> Coord:
    return (coord[0] - bounds[0], coord[1] - bounds[2], coord[2] - bounds[4])


def _overlap(shape, bounds):
    """Source and target slices for the part of ``bounds`` that lies inside ``shape``."""
    src, dst = [], []
    for axis in range(3):
        b0, b1 = bounds[2 * axis], bounds[2 * axis + 1]
        s0, s1 = max(b0, 0), min(b1, shape[axis])
        if s0 >= s1:
            return None
        src.append(slice(s0, s1))
        dst.append(slice(s0 - b0, s1 - b0))
    return tuple(src), tuple(dst)


def crop_and_pad(volume: np.ndarray, bounds: Bounds, pad_value=0.0) -> np.ndarray:
    d0, d1, h0, h1, w0, w1 = bounds
    out = np.full((d1 - d0, h1 - h0, w1 - w0), pad_value, dtype=volume.dtype)
    overlap = _overlap(volume.shape, bounds)
    if overlap:
        out[overlap[1]] = volume[overlap[0]]
    return out


def place_roi_into_volume(volume_shape, bounds: Bounds, roi_array: np.ndarray, fill_value=0) -> np.ndarray:
    out = np.full(volume_shape, fill_value, dtype=roi_array.dtype)
    overlap = _overlap(volume_shape, bounds)
    if overlap:
        out[overlap[0]] = roi_array[overlap[1]]
    return out


# ---------------------------------------------------------------- measurements
@dataclass
class SegmentMeasurements:
    voxel_count: int
    volume_mm3: float
    longest_diameter_mm: float

    @property
    def volume_cm3(self) -> float:
        return self.volume_mm3 / 1000.0


def measure_mask(mask: np.ndarray, spacing_xyz) -> Optional[SegmentMeasurements]:
    """Volume and longest linear extent of a (D, H, W) mask; spacing is (sx, sy, sz) mm."""
    from scipy.spatial import ConvexHull
    voxel_count = int(np.count_nonzero(mask))
    if voxel_count == 0:
        return None
    sx, sy, sz = spacing_xyz
    idx = np.argwhere(mask)
    physical = np.stack([idx[:, 2] * sx, idx[:, 1] * sy, idx[:, 0] * sz], axis=1)
    candidates = physical
    if len(physical) > 3:
        try:  # the farthest pair is always on the convex hull
            candidates = physical[ConvexHull(physical).vertices]
        except Exception:  # degenerate (flat) mask: subsample to bound the O(n^2) search
            if len(physical) > 4000:
                candidates = physical[np.random.default_rng(0).choice(len(physical), 4000, replace=False)]
    longest = 0.0
    if len(candidates) > 1:
        diffs = candidates[:, None, :] - candidates[None, :, :]
        longest = float(np.sqrt((diffs ** 2).sum(-1)).max())
    return SegmentMeasurements(voxel_count, float(voxel_count * sx * sy * sz), longest)


# ---------------------------------------------------------------- prompts / state
def sample_scribble(mask: np.ndarray, stride: int, max_points: int) -> List[Coord]:
    """Thin a painted (D, H, W) scribble mask to at most one point per ``stride``^3 cell.

    Each point is its cell's centre (clamped into the volume), so already-sent points stay
    put when more strokes are painted -- the engine then only feeds the new cells through.
    """
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return []
    stride = max(1, int(stride))
    cells = np.unique(idx // stride, axis=0)
    centres = np.minimum(cells * stride + stride // 2, np.array(mask.shape) - 1)
    if len(centres) > max_points:
        centres = centres[np.linspace(0, len(centres) - 1, max_points).round().astype(int)]
    return [tuple(int(v) for v in c) for c in centres]


@dataclass
class Prompts:
    points: List[PointPrompt] = field(default_factory=list)
    box: Optional[Tuple[Coord, Coord]] = None  # (min_corner, max_corner)
    text: Optional[str] = None

    @property
    def hasSpatial(self) -> bool:
        return bool(self.points) or self.box is not None


@dataclass
class SegmentState:
    """Per-segment inference state carried between runs."""
    roi_bounds: Optional[Bounds] = None
    prev_low_res_mask: object = None
    prev_full_mask: object = None
    image_embedding: object = None
    sent_points: List[PointPrompt] = field(default_factory=list)
    last_prob_roi: Optional[np.ndarray] = None
    last_uncertainty_roi: Optional[np.ndarray] = None
    full_mask: Optional[np.ndarray] = None
    threshold: float = 0.5
    iteration: int = 0

    def reset_state(self):
        self.roi_bounds = None
        self.prev_low_res_mask = None
        self.prev_full_mask = None
        self.image_embedding = None
        self.sent_points = []
        self.last_prob_roi = None
        self.last_uncertainty_roi = None

    def snapshot(self):
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()}

    def restore(self, snapshot):
        self.__dict__.update(snapshot)


# ---------------------------------------------------------------- engine
class InferenceEngine:
    def __init__(self, sam, critic, device, roi_target=128, roi_margin=8):
        self.sam, self.critic, self.device = sam, critic, device
        self.roi_target, self.roi_margin = roi_target, roi_margin

    def _prompts_fit(self, prompts: Prompts, bounds: Bounds) -> bool:
        coords = [c for c, _ in prompts.points] + (list(prompts.box) if prompts.box else [])
        return all(bounds_contains(bounds, c) for c in coords)

    def run(self, volume_norm: np.ndarray, state: SegmentState, prompts: Prompts) -> Tuple[np.ndarray, bool]:
        """Run/refine ``state`` with ``prompts``. Returns (full-volume binary mask, roi_tightened)."""
        import torch
        import torch.nn.functional as F
        if not prompts.hasSpatial:
            raise ValueError("Add at least one point, scribble or box first -- text alone can't localise a lesion.")
        device = self.device
        target = self.roi_target
        t4 = target // 4

        current = list(dict.fromkeys(prompts.points))
        currentSet = set(current)
        removed = any(p not in currentSet for p in state.sent_points)
        tightened = False
        if state.roi_bounds is None or removed or not self._prompts_fit(prompts, state.roi_bounds):
            bounds, tightened = compute_roi([c for c, _ in current], prompts.box, target, self.roi_margin)
            state.reset_state()
            state.roi_bounds = bounds
        bounds = state.roi_bounds
        sentSet = set(state.sent_points)
        new_points = [p for p in current if p not in sentSet]

        amp = torch.amp.autocast("cuda") if "cuda" in str(device) else contextlib.nullcontext()
        with torch.no_grad(), amp:
            if state.image_embedding is None:
                roi_image = crop_and_pad(volume_norm, bounds)
                image = torch.as_tensor(roi_image, dtype=torch.float32, device=device)[None, None]
                state.image_embedding = self.sam.image_encoder(image)
                del image

            text_arg = [prompts.text] if prompts.text else None
            box_arg = None
            if prompts.box is not None:
                corners = [to_model_order(to_local(c, bounds)) for c in prompts.box]
                box_arg = torch.tensor([corners], dtype=torch.float32, device=device)

            if state.prev_low_res_mask is None:
                low_res_mask = torch.zeros((1, 1, t4, t4, t4), dtype=torch.float32, device=device)
                low_res_conf = torch.zeros_like(low_res_mask)
                full_res_mask = None
                batches = [new_points] if new_points else [[]]
            else:
                low_res_mask = state.prev_low_res_mask
                low_res_conf = None
                full_res_mask = state.prev_full_mask
                batches = [[p] for p in new_points] if new_points else [[]]

            for batch in batches:
                if low_res_conf is None:
                    conf_map = (torch.sigmoid(self.critic(torch.sigmoid(full_res_mask).float())) > 0.5).float()
                    low_res_conf = F.interpolate(conf_map, size=(t4, t4, t4))
                points_arg = None
                if batch:
                    coords = torch.tensor([[to_model_order(to_local(c, bounds)) for c, _ in batch]],
                                          dtype=torch.float32, device=device)
                    labels = torch.tensor([[1 if positive else 0 for _, positive in batch]],
                                          dtype=torch.long, device=device)
                    points_arg = (coords, labels)
                sparse, dense = self.sam.prompt_encoder(
                    points=points_arg, boxes=box_arg, masks=low_res_mask, conf=low_res_conf, text=text_arg)
                low_res_mask, _ = self.sam.mask_decoder(
                    image_embeddings=state.image_embedding,
                    image_pe=self.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                )
                full_res_mask = F.interpolate(low_res_mask, size=(target,) * 3, mode="trilinear", align_corners=False)
                low_res_conf = None  # recomputed from the new mask for the next point, if any

            state.prev_low_res_mask = low_res_mask.detach()
            state.prev_full_mask = full_res_mask.detach()
            state.sent_points = current
            state.last_prob_roi = torch.sigmoid(full_res_mask).squeeze().float().cpu().numpy()
            # The critic's read on the mask actually being shown (P(voxel is an error)).
            state.last_uncertainty_roi = torch.sigmoid(
                self.critic(torch.sigmoid(full_res_mask).float())).squeeze().float().cpu().numpy()

        state.full_mask = self.binarize(volume_norm.shape, state)
        return state.full_mask, tightened

    @staticmethod
    def binarize(volume_shape, state: SegmentState) -> Optional[np.ndarray]:
        """Full-volume mask from the cached probabilities at ``state.threshold`` (no forward pass)."""
        if state.last_prob_roi is None or state.roi_bounds is None:
            return None
        roi_mask = (state.last_prob_roi > state.threshold).astype(np.uint8)
        return place_roi_into_volume(volume_shape, state.roi_bounds, roi_mask)

    @staticmethod
    def uncertainty_map(volume_shape, state: SegmentState) -> Optional[np.ndarray]:
        """Full-volume relative uncertainty in [0, 1] (NaN = not scored), or None if never run.

        The critic is an error detector (higher = more likely wrong) with very little absolute
        dynamic range, so it is min-max rescaled within the predicted region (at the model's own
        0.5 boundary, independent of the display threshold) dilated by 6 voxels; deep background
        is not scored. See SAT3D Studio's InferenceEngine.get_uncertainty_map for the rationale.
        """
        from scipy import ndimage
        if state.last_uncertainty_roi is None or state.roi_bounds is None or state.last_prob_roi is None:
            return None
        relevant = ndimage.binary_dilation(state.last_prob_roi > 0.5, iterations=6)
        if not relevant.any():
            return None
        u = state.last_uncertainty_roi
        normalized = np.full_like(u, np.nan)
        lo, hi = float(u[relevant].min()), float(u[relevant].max())
        normalized[relevant] = (u[relevant] - lo) / (hi - lo) if hi - lo >= 1e-8 else 0.5
        return place_roi_into_volume(volume_shape, state.roi_bounds, normalized, fill_value=np.nan)

import os
import random
import shutil
import sys
import contextlib
from datetime import datetime

import SimpleITK as sitk
import numpy as np
import qt
import slicer
import vtk

from slicer.ScriptedLoadableModule import *
from segment_anything_with_swin_conf2.build_samswin3D import sam_model_registry3D
from networks import Discriminator

# deps
try:
    import timm, monai, einops, torchio as tio
except ModuleNotFoundError:
    if slicer.util.confirmOkCancelDisplay("This module requires some Python packages. Click OK to install now."):
        slicer.util.pip_install("einops")
        slicer.util.pip_install("monai")
        slicer.util.pip_install("typer")
        slicer.util.pip_install("timm")
        slicer.util.pip_install("torchio")
        slicer.util.pip_install("medpy")
        slicer.util.pip_install("h5py")
        slicer.util.pip_install("yacs")
        slicer.util.pip_install("matplotlib")

import torch.nn.functional as F
from monai.data import decollate_batch

# your local sliding window
from .utils_monai_bts import sliding_window_inference


VAL_AMP = True


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.__stdout__
        self.log = open(logfile, "a")
    def write(self, message):
        self.terminal.write(message); self.log.write(message)
    def flush(self):
        self.terminal.flush(); self.log.flush()


class sat3DLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self._parameterNode = self.getParameterNode()

        self.download_location = qt.QStandardPaths.writableLocation(qt.QStandardPaths.DownloadLocation)
        self.sam, self.critic, self.device = None, None, None
        self.torch = None

        # I/O + sizing
        self.image_size = 128

        # prompts
        self.include_coords = {}
        self.exclude_coords = {}
        self._prev_include_set = set()
        self._prev_exclude_set = set()

        # state
        self.slice_direction = 'Red'
        self.dimension = 3

        # outputs
        self.mask = np.zeros((1, 1, 1))
        self.mask_backup = None
        self.iteration = 0
        self.save_points = None
        self.save_labels = None
        self.current_task_name = None
        self.out_dir = None
        self.log_file_name = None

        self.mask_locations = set()
        self.interp_slice_direction = set()

        # cache for refinement (torch on device, keep fp16 on CUDA)
        self.visible_prev_mask = None
        self.prev_mask_ = None

        # TorchIO masked z-norm (use foreground > 0)
        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

        # ---- quality & memory knobs ----
        self.USE_CRITIC_CONF = True   # toggle critic gating
        self.OVERLAP = 0.625          # smoother tiling
        self.THRESH = 0.5             # binarization threshold
        self.PCT_CLIP = (0.5, 99.5)   # pre-norm robust clipping

        # cache image shape for bounds checks
        self.img = np.zeros((1, 1, 1))  # (D,H,W)

    # ---------- util ----------
    @staticmethod
    def remove_module_prefix(state_dict):
        return { (k.replace("module.", "") if k.startswith("module.") else k): v for k, v in state_dict.items() }

    @staticmethod
    def postprocess_masks_like_script(low_res_masks, image_size, original_size, ft2d=False):
        masks = F.interpolate(low_res_masks, (image_size, image_size, image_size),
                              mode="trilinear", align_corners=False)
        if ft2d and min(original_size) < image_size:
            raise NotImplementedError
        else:
            masks = F.interpolate(masks, original_size, mode="trilinear", align_corners=False)
        return masks, None

    def set_current_case_name(self, name: str):
        """Store a human-readable case name for output folders and saved files."""
        self.current_task_name = name

    # ---------- torch setup ----------
    def setupPythonRequirements(self):
        try:
            import PyTorchUtils
        except ModuleNotFoundError:
            slicer.util.errorDisplay("This module requires the PyTorch extension. Install it from Extensions Manager.")
            return False

        minimumTorchVersion = "2.1"
        minimumTorchVisionVersion = "0.16"
        torchLogic = PyTorchUtils.PyTorchUtilsLogic()
        if not torchLogic.torchInstalled():
            slicer.util.delayDisplay("Installing PyTorch (may take several minutes)...")
            torch_inst = torchLogic.installTorch(
                askConfirmation=True,
                torchVersionRequirement=f">={minimumTorchVersion}",
                torchvisionVersionRequirement=f">={minimumTorchVisionVersion}",
                forceComputationBackend='cu121'
            )
            if torch_inst is None:
                raise ValueError('PyTorch extension needs to be installed to use this module.')
        else:
            from packaging import version
            if version.parse(torchLogic.torch.__version__) < version.parse(minimumTorchVersion):
                raise ValueError(
                    f'PyTorch {torchLogic.torch.__version__} < {minimumTorchVersion}. '
                    f'Use "PyTorch Util" to install a compatible version.'
                )
        self.torch = torchLogic.importTorch()
        try:
            import timm, monai, einops  # noqa
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay("Extra Python packages required. Install now?"):
                slicer.util.pip_install("einops monai typer timm torchio medpy h5py yacs matplotlib")
        return True

    def _empty_cache(self):
        if self.device and "cuda" in str(self.device):
            try: self.torch.cuda.empty_cache()
            except Exception: pass

    # ---------- model load ----------
    def create_sam(self, sam_weights_path, sam_critic_weights_path, modeltype, seed, log_fname):
        seed = int(seed)
        self.log_file_name = log_fname
        slicer.util.delayDisplay(f"Loading SAT3D (seed={seed}) ... ")

        if not self.setupPythonRequirements():
            return

        try:
            self.sam = sam_model_registry3D[modeltype](checkpoint=None)
            self.critic = Discriminator()
        except FileNotFoundError:
            slicer.util.infoDisplay("SAT3D weights not found, use Download button")
            self._parameterNode.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR_IN_LOCATING_WEIGHTS")
            return

        if self.torch.cuda.is_available():
            self.device = "cuda:0"
            self.sam.to(device="cuda"); self.critic.to(device="cuda")
            self.torch.cuda.manual_seed(seed)
        else:
            self.device = "cpu"
            self.torch.manual_seed(seed)
        random.seed(seed); np.random.seed(seed)

        model_dict  = self.torch.load(sam_weights_path, map_location=self.device, weights_only=False)
        state_dict  = self.remove_module_prefix(model_dict['model_state_dict'])
        self.sam.load_state_dict(state_dict, strict=False)
        del model_dict, state_dict

        c_model_dict = self.torch.load(sam_critic_weights_path, map_location=self.device, weights_only=False)
        c_state_dict = self.remove_module_prefix(c_model_dict['model_state_dict'])
        self.critic.load_state_dict(c_state_dict, strict=False)
        del c_model_dict, c_state_dict

        self.sam.eval(); self.critic.eval()
        for p in self.sam.parameters(): p.requires_grad_(False)
        for p in self.critic.parameters(): p.requires_grad_(False)

        self._empty_cache()
        self._parameterNode.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] MODEL-WEIGHTS-LOADED")

    # ---------- geometry ----------
    def generateaffine(self):
        """Use full IJK->RAS matrix and cache its inverse (RAS->IJK)."""
        vol = self._parameterNode.GetNodeReference("fastsamInputVolume")
        self.img = slicer.util.arrayFromVolume(vol)  # (D,H,W) for bounds checks

        ijkToRAS = vtk.vtkMatrix4x4()
        vol.GetIJKToRASMatrix(ijkToRAS)
        self.affine = slicer.util.arrayFromVTKMatrix(ijkToRAS).astype(np.float32)

        # Cache inverse for fast RAS->IJK mapping
        self.ras_to_ijk = np.linalg.inv(self.affine).astype(np.float32)

        self.origin = vol.GetOrigin()
        self.spacing = vol.GetSpacing()

    # ---------- slicer visualization ----------
    def pass_mask_to_slicer(self):
        segmentationNode = self._parameterNode.GetNodeReference("fastsamSegmentation")
        volumeNode = self._parameterNode.GetNodeReference("fastsamInputVolume")
        segmentation = segmentationNode.GetSegmentation()

        if hasattr(segmentation, "SetMasterRepresentation"):
            segmentation.SetMasterRepresentation('Binary labelmap')

        defaultID = segmentation.GetSegmentIdBySegmentName("Segment_1")
        if defaultID: segmentation.RemoveSegment(defaultID)

        SEG_CURRENT, SEG_ADDED, SEG_REMOVED = "Tumor (current)", "Δ Added", "Δ Removed"
        color_current = (1.0, 215/255.0, 0.0); color_added = (56/255.0, 163/255.0, 63/255.0); color_removed = (145/255.0, 60/255.0, 66/255.0)

        segs = self.mask  # (3,D,H,W)
        cur_bin = (segs[0] > 0.5)

        labelmap = np.zeros(cur_bin.shape, dtype=np.uint8); labelmap[cur_bin] = 1
        self.label_map = labelmap

        def get_or_add(segmentation, name, color):
            seg = segmentation.GetSegment(segmentation.GetSegmentIdBySegmentName(name))
            if seg is None:
                seg_id = segmentation.AddEmptySegment(name); seg = segmentation.GetSegment(seg_id)
            else:
                seg_id = segmentation.GetSegmentIdBySegmentName(name)
            seg.SetColor(*color); return seg_id

        seg_id_current = get_or_add(segmentation, SEG_CURRENT, color_current)
        slicer.util.updateSegmentBinaryLabelmapFromArray((labelmap == 1).astype(np.uint8), segmentationNode, seg_id_current, volumeNode)

        if self.iteration != 0 and self.prev_mask_ is not None:
            prev_bin = (self.prev_mask_[0] > 0.5)
            delta_added   = np.logical_and(cur_bin, np.logical_not(prev_bin)).astype(np.uint8)
            delta_removed = np.logical_and(prev_bin, np.logical_not(cur_bin)).astype(np.uint8)
            self._parameterNode.logger.info(f"[Diff] cur={int(cur_bin.sum())} added={int(delta_added.sum())} removed={int(delta_removed.sum())}")

            if delta_added.any():
                seg_id_added = segmentation.GetSegmentIdBySegmentName(SEG_ADDED) or segmentation.AddEmptySegment(SEG_ADDED)
                segmentation.GetSegment(seg_id_added).SetColor(*color_added)
                slicer.util.updateSegmentBinaryLabelmapFromArray(delta_added, segmentationNode, seg_id_added, volumeNode)
            else:
                sid = segmentation.GetSegmentIdBySegmentName(SEG_ADDED)
                if sid: slicer.util.updateSegmentBinaryLabelmapFromArray(np.zeros_like(delta_added, dtype=np.uint8), segmentationNode, sid, volumeNode)

            if delta_removed.any():
                old_id = segmentation.GetSegmentIdBySegmentName(SEG_REMOVED)
                if old_id: segmentation.RemoveSegment(old_id)
                seg_id_removed = segmentation.AddEmptySegment(SEG_REMOVED)
                segmentation.GetSegment(seg_id_removed).SetColor(*color_removed)
                slicer.util.updateSegmentBinaryLabelmapFromArray(delta_removed, segmentationNode, seg_id_removed, volumeNode)
            else:
                sid = segmentation.GetSegmentIdBySegmentName(SEG_REMOVED)
                if sid: slicer.util.updateSegmentBinaryLabelmapFromArray(np.zeros_like(delta_removed, dtype=np.uint8), segmentationNode, sid, volumeNode)
        else:
            for name in (SEG_ADDED, SEG_REMOVED):
                sid = segmentation.GetSegmentIdBySegmentName(name)
                if sid: slicer.util.updateSegmentBinaryLabelmapFromArray(np.zeros_like(labelmap, dtype=np.uint8), segmentationNode, sid, volumeNode)

        segmentationNode.CreateDefaultDisplayNodes()
        dispNode = segmentationNode.GetDisplayNode()
        if dispNode:
            if hasattr(dispNode, "SetPreferredDisplayRepresentationName2D"):
                dispNode.SetPreferredDisplayRepresentationName2D("Binary labelmap")
            if hasattr(dispNode, "SetVisibility2D"): dispNode.SetVisibility2D(True)
            if hasattr(dispNode, "SetVisibility2DFill"): dispNode.SetVisibility2DFill(True)
            if hasattr(dispNode, "SetVisibility2DOutline"): dispNode.SetVisibility2DOutline(False)
            if hasattr(dispNode, "SetSmoothingFactor"): dispNode.SetSmoothingFactor(0.0)
            if hasattr(dispNode, "SetPreferredDisplayRepresentationName3D"):
                dispNode.SetPreferredDisplayRepresentationName3D('Closed surface')

        segmentationNode.CreateClosedSurfaceRepresentation()
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        slicer.util.resetThreeDViews()

        self._parameterNode.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SEGMENTS-LOADED")
        self.prev_mask_ = self.mask.copy()

    # ---------- saving ----------
    def maybeSaveSegmentation(self):
        case_name = self.current_task_name or "Case"
        output_path = os.path.join(self.download_location, case_name)
        self._parameterNode.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CREATED-PATH: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        self.out_dir = output_path

        if self.save_points is not None and self.save_labels is not None:
            pts = self.save_points.detach().cpu().numpy(); lbs = self.save_labels.detach().cpu().numpy()
            points_file_name = os.path.join(output_path, f"{case_name}_{self.iteration}_{datetime.now().strftime('%Y-%m-%d %H%M%S')}_points.txt")
            with open(points_file_name, 'w') as f:
                for pt, label in zip(pts.squeeze(0), lbs.squeeze(0)):
                    coord_str = ', '.join(f'{x:.1f}' for x in pt); f.write(f'{coord_str}; {int(label)}\n')
            del pts, lbs
            self.save_points = None; self.save_labels = None

        file_name = f"{case_name}_{self.iteration}_{datetime.now().strftime('%Y-%m-%d %H%M%S')}.nii.gz"
        self.saveSegmentationAsNifti(output_path=output_path, file_name=file_name)

    def saveSegmentationAsNifti(self, output_path=None, file_name=None):
        if not hasattr(self, "label_map") or self.label_map is None:
            slicer.util.errorDisplay("Label map not found. Run segmentation first."); return
        out_path = os.path.join(output_path, file_name)
        flipped = np.flip(self.label_map, axis=(1, 2))
        sitk_mask = sitk.GetImageFromArray(flipped.astype(np.uint8))

        refVolumeNode = self._parameterNode.GetNodeReference("fastsamInputVolume")
        spacing = refVolumeNode.GetSpacing(); origin = refVolumeNode.GetOrigin()
        ijkToRAS = vtk.vtkMatrix4x4(); refVolumeNode.GetIJKToRASMatrix(ijkToRAS)
        direction = [ijkToRAS.GetElement(row, col) / spacing[row] for col in range(3) for row in range(3)]

        sitk_mask.SetSpacing(spacing); sitk_mask.SetOrigin(origin); sitk_mask.SetDirection(direction)
        sitk.WriteImage(sitk_mask, out_path)
        self._parameterNode.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] NIFTI-SAVED: {out_path}")
        slicer.util.infoDisplay(f"Segmentation saved to: {out_path}")
        del sitk_mask, flipped, direction
        self._empty_cache()

    def endRefinementTask(self):
        self._parameterNode.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] END-TASK")
        log_file_path = os.path.join(self.download_location, self.log_file_name)
        if getattr(self, "out_dir", None):
            try: shutil.copy2(log_file_path, self.out_dir)
            except Exception: pass
        slicer.util.infoDisplay("TASK-COMPLETED")
        self._empty_cache(); slicer.app.restart()

    # ---------- inference ----------
    def get_volume_node(self):
        return self._parameterNode.GetNodeReference("fastsamInputVolume")

    def inference(self, input, model, patch_size, low_res_prev_masks, points, low_res_conf):
        torch = self.torch
        use_amp = bool(VAL_AMP and (self.device and "cuda" in str(self.device)))
        ctx = (torch.amp.autocast('cuda') if use_amp else contextlib.nullcontext())
        with torch.inference_mode():
            with ctx:
                return sliding_window_inference(
                    inputs=input,
                    roi_size=patch_size,
                    sw_batch_size=1,
                    predictor=model,
                    points=points,
                    low_res_prev_masks=low_res_prev_masks,
                    overlap=self.OVERLAP,
                    low_res_conf=low_res_conf
                )

    def get_mask(self):
        try:
            vol_node = self.get_volume_node()
            if vol_node is None:
                self._parameterNode.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PREDICT-FAILED: No input volume.")
                return
            if self.current_task_name is None:
                self.current_task_name = vol_node.GetName()

            # -------- data prep: clip + TorchIO ZNorm --------
            arr = slicer.util.arrayFromVolume(vol_node)  # (D,H,W)
            fg = arr > 0
            if np.any(fg):
                lo, hi = np.percentile(arr[fg], self.PCT_CLIP)
                arr = np.clip(arr, lo, hi)
            inputimage = arr[np.newaxis, np.newaxis, ...]  # (1,1,D,H,W)

            torch = self.torch
            inputimage_tensor = torch.as_tensor(inputimage, dtype=torch.float32)

            # TorchIO expects (C, D, H, W); squeeze/add back batch
            tio_img = tio.ScalarImage(tensor=inputimage_tensor.squeeze(0))
            tio_img = self.norm_transform(tio_img)
            inputimage_tensor = tio_img.data.unsqueeze(0)  # (1,1,D,H,W)
            del tio_img

            # ---- shapes and device move ----
            roi_size = (self.image_size, self.image_size, self.image_size)
            input_pv = torch.empty(inputimage.shape[0], 1, inputimage.shape[2], inputimage.shape[3], inputimage.shape[4])

            if self.device and "cuda" in str(self.device):
                inputimage_tensor = inputimage_tensor.pin_memory().to(self.device, non_blocking=True)
            else:
                inputimage_tensor = inputimage_tensor.to(self.device)

            # ---- prompts: compute true deltas ----
            def _to_tuple_int(p): return (int(p[0]), int(p[1]), int(p[2]))
            cur_inc_set = set(_to_tuple_int(c) for c in self.include_coords.values()) if self.include_coords else set()
            cur_exc_set = set(_to_tuple_int(c) for c in self.exclude_coords.values()) if self.exclude_coords else set()
            new_inc = list(cur_inc_set) # - self._prev_include_set)
            new_exc = list(cur_exc_set) #- self._prev_exclude_set)
            do_initial = (self.iteration == 0 and not cur_inc_set and not cur_exc_set)

            # ---- low-res state ----
            if do_initial:
                prev_masks = torch.zeros_like(input_pv, device=self.device)
                prev_low_res_mask = F.interpolate(prev_masks.float(), size=(roi_size[0] // 4, roi_size[0] // 4, roi_size[0] // 4))
                if self.USE_CRITIC_CONF:
                    conf_map = torch.sigmoid(self.critic(torch.sigmoid(prev_masks).float()))
                    low_res_conf = F.interpolate(conf_map, size=(roi_size[0] // 4, roi_size[0] // 4, roi_size[0] // 4))
                    del conf_map
                else:
                    low_res_conf = torch.zeros_like(prev_low_res_mask)

                output = self.inference(
                    inputimage_tensor, self.sam, roi_size,
                    low_res_prev_masks=prev_low_res_mask,
                    points=None,
                    low_res_conf=low_res_conf
                )
                del prev_masks, prev_low_res_mask, low_res_conf
                self._parameterNode.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AI-PRED: initial")
            else:
                if (new_inc or new_exc):
                    points_np = np.array(new_inc + new_exc, dtype=np.int32)
                    labels_np = np.array([1] * len(new_inc) + [0] * len(new_exc), dtype=np.int64)
                    points_t  = torch.from_numpy(points_np).to(self.device).unsqueeze(0)
                    labels_t  = torch.from_numpy(labels_np).to(self.device).unsqueeze(0)
                    self.save_points, self.save_labels = points_t, labels_t
                else:
                    points_t, labels_t = None, None

                prev_lr_src = self.visible_prev_mask if self.visible_prev_mask is not None \
                              else torch.zeros_like(input_pv, device=self.device)
                prev_low_res_mask = F.interpolate(prev_lr_src.float(), size=(roi_size[0] // 4, roi_size[0] // 4, roi_size[0] // 4))

                if self.USE_CRITIC_CONF:
                    conf_map = torch.sigmoid(self.critic(torch.sigmoid(prev_lr_src).float()))
                    low_res_conf = F.interpolate(conf_map, size=(roi_size[0] // 4, roi_size[0] // 4, roi_size[0] // 4))
                    del conf_map
                else:
                    low_res_conf = torch.zeros_like(prev_low_res_mask)

                self._parameterNode.logger.info(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DELTA-POINTS: +inc={len(new_inc)} +exc={len(new_exc)}"
                )
                output = self.inference(
                    inputimage_tensor, self.sam, roi_size,
                    low_res_prev_masks=prev_low_res_mask,
                    points=[points_t, labels_t] if points_t is not None else None,
                    low_res_conf=low_res_conf
                )
                del prev_lr_src, prev_low_res_mask, low_res_conf
                self._parameterNode.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AI-PRED: delta")

            # ---- postproc ----
            out_list = decollate_batch(output)
            # cache prev mask for next round (torch)
            self.visible_prev_mask = out_list[0].unsqueeze(0)
            if self.device and "cuda" in str(self.device):
                self.visible_prev_mask = self.visible_prev_mask.to(self.device, dtype=self.torch.float16, non_blocking=True)
            else:
                self.visible_prev_mask = self.visible_prev_mask.to(self.device)

            # sigmoid -> threshold once; CPU to free VRAM
            prob = self.torch.sigmoid(out_list[0])
            mask_np = (prob.detach().cpu().numpy() > self.THRESH).astype(np.uint8)

            # ensure (3,D,H,W), ch0 current
            seg_3 = np.zeros((3,) + mask_np.shape[-3:], dtype=np.uint8)
            seg_3[0] = mask_np.squeeze()

            # free
            del out_list, output, inputimage_tensor, inputimage, arr, mask_np, prob
            self._empty_cache()

            self.mask = seg_3
            self.pass_mask_to_slicer()
            self.maybeSaveSegmentation()

            # history & iter
            self._prev_include_set = cur_inc_set
            self._prev_exclude_set = cur_exc_set
            self.iteration += 1

        finally:
            self._empty_cache()

    # ---------- misc ----------
    def backup_mask(self):
        self.mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            self._parameterNode.GetNodeReference("fastsamSegmentation"),
            self._parameterNode.GetParameter("fastsamCurrentSegment"))

    def undo(self):
        if self.mask_backup is not None:
            self.mask = self.mask_backup.copy()
            self.pass_mask_to_slicer()

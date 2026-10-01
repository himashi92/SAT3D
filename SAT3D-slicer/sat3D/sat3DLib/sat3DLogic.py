"""Model loading, inference orchestration, display and I/O for the SAT3D Slicer module.

Heavy dependencies (PyTorch, TorchIO, transformers, the model code) are imported lazily in
``ensureDependencies`` so the module still loads in a fresh Slicer and can offer to install them.
The model itself runs in ``engine.InferenceEngine`` (ported from SAT3D Studio), on a worker
thread so Slicer stays responsive; everything touching MRML stays on the main thread.
"""
import importlib.util
import json
import logging
import os
import random
import threading
import time
from datetime import datetime

import numpy as np
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

from .engine import InferenceEngine, Prompts, SegmentState, measure_mask, preprocess_volume

logger = logging.getLogger("SAT3D")

# Parameter node references / parameters (names kept for compatibility with saved scenes)
INPUT_VOLUME_REF = "fastsamInputVolume"
INCLUDE_POINTS_REF = "fastsamIncludePoints"
EXCLUDE_POINTS_REF = "fastsamExcludePoints"
SEGMENTATION_REF = "fastsamSegmentation"
CURRENT_SEGMENT_PARAM = "fastsamCurrentSegment"
BOX_REF = "sat3dBox"
SCRIBBLE_SEGMENTATION_REF = "sat3dScribbles"
UNCERTAINTY_VOLUME_REF = "sat3dUncertainty"

DEFAULT_SEGMENT_NAME = "Tumor"
DEFAULT_SEGMENT_COLOR = (1.0, 215 / 255.0, 0.0)
SEG_ADDED, SEG_REMOVED = "Δ Added", "Δ Removed"
DELTA_COLORS = {SEG_ADDED: (56 / 255.0, 163 / 255.0, 63 / 255.0), SEG_REMOVED: (145 / 255.0, 60 / 255.0, 66 / 255.0)}
DELTA_SEGMENT_NAMES = tuple(DELTA_COLORS)
SCRIBBLE_POS, SCRIBBLE_NEG = "Scribble +", "Scribble −"
APPROVED_TAG = "SAT3D.Approved"

# pip requirements of the model / inference code (torch itself is handled by PyTorchUtils)
REQUIRED_PACKAGES = {"einops": "einops", "timm": "timm", "torchio": "torchio",
                     "transformers": "transformers>=4.40,<4.46"}  # newer transformers needs torch>=2.2
MIN_TORCH_VERSION = "2.1"
MIN_TORCHVISION_VERSION = "0.16"
TORCH_COMPUTATION_BACKEND = "cu121"  # set to None to let PyTorchUtils auto-detect

UNDO_DEPTH = 5
MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOADS = qt.QStandardPaths.writableLocation(qt.QStandardPaths.DownloadLocation)


class Settings:
    """User preferences, persisted in Slicer's settings under ``SAT3D/``."""
    DEFAULTS = {
        "samCheckpoint": os.path.join(DOWNLOADS, "sam_model_dice_best.pth"),
        "criticCheckpoint": os.path.join(DOWNLOADS, "critic_dice_best.pth"),
        "textEncoderDir": os.path.join(MODULE_DIR, "Resources", "TextEncoder"),
        "device": "auto",           # auto | cuda | cpu
        "seed": 2025,
        "roiMargin": 8,             # voxels around the prompts when choosing the 128^3 crop
        "scribbleStride": 3,        # one point per stride^3 painted cell
        "maxScribblePoints": 200,
        "boxDepth": 20,             # slices, for boxes drawn flat in one view
        "autoSaveRuns": True,       # write each run's mask to the session folder
    }

    def __getattr__(self, name):
        if name not in Settings.DEFAULTS:
            raise AttributeError(name)
        default = Settings.DEFAULTS[name]
        value = slicer.app.userSettings().value(f"SAT3D/{name}", default)
        if isinstance(default, bool):
            return value if isinstance(value, bool) else str(value).lower() == "true"
        return type(default)(value)

    def set(self, name, value):
        slicer.app.userSettings().setValue(f"SAT3D/{name}", value)


def worldToVoxel(volumeNode, worldPos):
    """World (RAS) position -> (d, h, w) voxel index of ``volumeNode`` (honours parent transforms)."""
    worldToVolume = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, volumeNode.GetParentTransformNode(), worldToVolume)
    ras = worldToVolume.TransformPoint(worldPos)
    rasToIjk = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(rasToIjk)
    i, j, k = rasToIjk.MultiplyPoint([*ras, 1.0])[:3]
    return int(round(k)), int(round(j)), int(round(i))


def volumeShape(volumeNode):
    w, h, d = volumeNode.GetImageData().GetDimensions()
    return d, h, w


def isVoxelInVolume(volumeNode, voxel):
    return all(0 <= v < n for v, n in zip(voxel, volumeShape(volumeNode)))


def saveMaskAsNifti(mask, referenceVolumeNode, path):
    """Save a (D, H, W) label array with the geometry of ``referenceVolumeNode`` (RAS -> LPS)."""
    import SimpleITK as sitk
    ijkToRas = vtk.vtkMatrix4x4()
    referenceVolumeNode.GetIJKToRASMatrix(ijkToRas)
    spacing = referenceVolumeNode.GetSpacing()
    rasToLps = (-1.0, -1.0, 1.0)
    image = sitk.GetImageFromArray(np.ascontiguousarray(mask, dtype=np.uint8))
    image.SetSpacing(spacing)
    image.SetOrigin([rasToLps[r] * ijkToRas.GetElement(r, 3) for r in range(3)])
    image.SetDirection([rasToLps[r] * ijkToRas.GetElement(r, c) / spacing[c] for r in range(3) for c in range(3)])
    sitk.WriteImage(image, path, useCompression=True)


def safeName(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


class sat3DLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.settings = Settings()
        self.torch = None
        self.engine = None
        self._logHandler = None
        self._job = None
        self.resetCase()

    # ---------- case / session ----------
    def resetCase(self, volumeNode=None):
        """Forget all refinement state. The session folder/log for ``volumeNode`` is opened on first use."""
        self.caseName = volumeNode.GetName() if volumeNode else None
        self._caseVolume = volumeNode
        self.segmentStates = {}
        self.undoStacks = {}
        self._normVolume = None
        self._normVolumeKey = None
        self._sessionDir = None
        self._closeSessionLog()
        self._emptyCache()

    @property
    def sessionDir(self):
        """``<scan folder>/<case>_sat3d_session`` (Downloads if the scan has no file), created with its log on
        first use -- resolved lazily because a just-loaded volume gets its storage node after it is selected."""
        if self._sessionDir is None and self._caseVolume is not None:
            storage = self._caseVolume.GetStorageNode()
            source = storage.GetFileName() if storage and storage.GetFileName() else None
            base = os.path.dirname(source) if source else DOWNLOADS
            self._sessionDir = os.path.join(base, f"{safeName(self.caseName)}_sat3d_session")
            os.makedirs(self._sessionDir, exist_ok=True)
            self._logHandler = logging.FileHandler(os.path.join(self._sessionDir, "session_log.txt"), encoding="utf-8")
            self._logHandler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
            logger.addHandler(self._logHandler)
            logger.setLevel(logging.INFO)
            logger.info(f"OPENED: {source or self.caseName} shape={volumeShape(self._caseVolume)}")
        return self._sessionDir

    def _closeSessionLog(self):
        if self._logHandler is not None:
            logger.removeHandler(self._logHandler)
            self._logHandler.close()
            self._logHandler = None

    def log(self, message):
        """Write to the current case's session log (opening it if needed)."""
        _ = self.sessionDir
        logger.info(message)

    def segmentState(self, segmentId):
        return self.segmentStates.setdefault(segmentId, SegmentState())

    def forgetSegment(self, segmentId):
        self.segmentStates.pop(segmentId, None)
        self.undoStacks.pop(segmentId, None)

    # ---------- dependencies ----------
    def ensureDependencies(self):
        """Install/import PyTorch and the other required packages. Returns False if unavailable."""
        if self.torch is not None:
            return True
        try:
            import PyTorchUtils
        except ModuleNotFoundError:
            slicer.util.errorDisplay("SAT3D requires the PyTorch extension. Install it from the Extensions Manager.")
            return False

        torchLogic = PyTorchUtils.PyTorchUtilsLogic()
        if not torchLogic.torchInstalled():
            torch = torchLogic.installTorch(
                askConfirmation=True,
                torchVersionRequirement=f">={MIN_TORCH_VERSION}",
                torchvisionVersionRequirement=f">={MIN_TORCHVISION_VERSION}",
                forceComputationBackend=TORCH_COMPUTATION_BACKEND,
            )
            if torch is None:
                slicer.util.errorDisplay("PyTorch is required to run SAT3D.")
                return False
        else:
            from packaging import version
            if version.parse(torchLogic.torch.__version__) < version.parse(MIN_TORCH_VERSION):
                slicer.util.errorDisplay(
                    f"PyTorch {torchLogic.torch.__version__} is older than the required {MIN_TORCH_VERSION}.\n"
                    f'Use the "PyTorch Util" module to install a compatible version.')
                return False

        missing = [req for module, req in REQUIRED_PACKAGES.items() if importlib.util.find_spec(module) is None]
        if missing:
            if not slicer.util.confirmOkCancelDisplay(
                    f"SAT3D requires these Python packages: {', '.join(missing)}.\nClick OK to install them now."):
                return False
            with slicer.util.WaitCursor():
                slicer.util.pip_install(" ".join(f'"{m}"' for m in missing))

        self.torch = torchLogic.importTorch()
        return True

    def _emptyCache(self):
        if self.engine is not None and "cuda" in str(self.engine.device):
            self.torch.cuda.empty_cache()

    # ---------- model ----------
    @property
    def modelLoaded(self):
        return self.engine is not None

    def unloadModel(self):
        self.engine = None
        for state in self.segmentStates.values():  # cached embeddings belong to the old model
            state.reset_state()
        self._emptyCache()
        self.log("MODEL-UNLOADED")

    def loadModel(self):
        """Build SAT3D-plus + critic and load their weights. Returns False if dependencies are unavailable."""
        if not self.ensureDependencies():
            return False
        s = self.settings
        for path in (s.samCheckpoint, s.criticCheckpoint):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model weights not found: {path}\nSet the paths under Advanced.")
        torch = self.torch
        from segment_anything_with_swin_conf_plus.build_samswin3D import sam_model_registry3D
        from networks import Discriminator

        torch.manual_seed(s.seed)
        random.seed(s.seed)
        np.random.seed(s.seed)
        device = s.device if s.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        textDir = s.textEncoderDir if os.path.isdir(s.textEncoderDir) else None
        if textDir:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        sam = self._loadWeights(sam_model_registry3D["swin2"](checkpoint=None, text_model_name=textDir), s.samCheckpoint)
        critic = self._loadWeights(Discriminator(), s.criticCheckpoint)
        for model in (sam, critic):
            model.to(device).eval()
            model.requires_grad_(False)

        self.engine = InferenceEngine(sam, critic, device, roi_target=128, roi_margin=s.roiMargin)
        for state in self.segmentStates.values():
            state.reset_state()
        self._emptyCache()
        self.log(f"MODEL-LOADED: device={self.deviceName()} seed={s.seed}")
        return True

    def _loadWeights(self, model, path):
        checkpoint = self.torch.load(path, map_location="cpu", weights_only=False)
        stateDict = {k.removeprefix("module."): v for k, v in checkpoint["model_state_dict"].items()}
        result = model.load_state_dict(stateDict, strict=False)
        if result.missing_keys:
            raise RuntimeError(f"{os.path.basename(path)} is missing {len(result.missing_keys)} weights, "
                               f"e.g. {result.missing_keys[:3]}. Is it the SAT3D-plus checkpoint?")
        if result.unexpected_keys:
            self.log(f"WEIGHTS: ignored {len(result.unexpected_keys)} unused keys in {os.path.basename(path)}")
        return model

    def deviceName(self):
        if self.engine is None:
            return "not loaded"
        device = str(self.engine.device)
        if "cuda" in device:
            try:
                return f"cuda ({self.torch.cuda.get_device_name(0)})"
            except Exception:
                return "cuda"
        return device

    # ---------- inference ----------
    def _normalizedVolume(self, volumeNode):
        key = (volumeNode.GetID(), volumeNode.GetImageData().GetMTime())
        if self._normVolumeKey != key:
            self._normVolume = None
            self._normVolumeKey = key
        return self._normVolume

    @property
    def busy(self):
        return self._job is not None and self._job["thread"].is_alive()

    def startPrediction(self, volumeNode, segmentId, prompts: Prompts):
        """Start a run on a worker thread. Poll ``pollPrediction`` from the main thread for the result."""
        if not self.modelLoaded:
            raise RuntimeError("SAT3D model is not loaded.")
        if self.busy:
            raise RuntimeError("A prediction is already running.")
        if not prompts.hasSpatial:
            raise ValueError("Add at least one point, scribble or box first -- text alone can't localise a lesion.")
        state = self.segmentState(segmentId)
        stack = self.undoStacks.setdefault(segmentId, [])
        stack.append(state.snapshot())
        del stack[:-UNDO_DEPTH]

        normVolume = self._normalizedVolume(volumeNode)
        rawVolume = None if normVolume is not None else slicer.util.arrayFromVolume(volumeNode).copy()
        job = {"segmentId": segmentId, "prompts": prompts, "start": time.time(),
               "result": None, "error": None, "previousMask": state.full_mask}

        def work():
            try:
                volume = normVolume
                if volume is None:
                    volume = preprocess_volume(rawVolume)
                    job["normVolume"] = volume
                job["result"] = self.engine.run(volume, state, prompts)
            except Exception as e:
                import traceback
                job["error"] = e
                job["traceback"] = traceback.format_exc()
            finally:
                self._emptyCache()

        job["thread"] = threading.Thread(target=work, name="SAT3D-inference", daemon=True)
        self._job = job
        job["thread"].start()

    def pollPrediction(self, volumeNode, segmentationNode, wait=0.08):
        """None while running; otherwise finishes the run on the main thread and returns the job dict.

        Waits up to ``wait`` seconds for the worker: Slicer's main thread holds the GIL while idle in the
        Qt event loop, so the worker only makes progress while the main thread is blocked here.
        """
        job = self._job
        if job is None:
            return None
        job["thread"].join(timeout=wait)
        if job["thread"].is_alive():
            return None
        self._job = None
        job["elapsed"] = time.time() - job["start"]
        segmentId, prompts = job["segmentId"], job["prompts"]
        if "normVolume" in job:
            self._normVolume = job["normVolume"]
        if job["error"] is not None:
            self.undo(None, None, segmentId, display=False)  # drop the snapshot taken for this run
            self.log(f"RUN-FAILED: {job['error']}")
            return job

        mask, tightened = job["result"]
        state = self.segmentState(segmentId)
        self.showSegment(volumeNode, segmentationNode, segmentId, mask, job["previousMask"])
        segmentName = segmentationNode.GetSegmentation().GetSegment(segmentId).GetName()
        if self.settings.autoSaveRuns:
            self._saveRun(volumeNode, segmentName, state.iteration, mask, prompts)
        self.log(f"RUN: segment={segmentName} points={len(prompts.points)} "
                    f"box={'yes' if prompts.box else 'no'} text={prompts.text} "
                    f"took={job['elapsed']:.1f}s tightened={tightened} voxels={int(mask.sum())}")
        state.iteration += 1
        job["tightened"] = tightened
        return job

    def rethreshold(self, volumeNode, segmentationNode, segmentId, threshold):
        state = self.segmentState(segmentId)
        state.threshold = threshold
        mask = InferenceEngine.binarize(volumeShape(volumeNode), state)
        if mask is not None:
            state.full_mask = mask
            self.showSegment(volumeNode, segmentationNode, segmentId, mask, None)
        return mask is not None

    def undo(self, volumeNode, segmentationNode, segmentId, display=True):
        """Restore the state before the last run of ``segmentId``. Returns False if there is nothing to undo."""
        stack = self.undoStacks.get(segmentId)
        if not stack:
            return False
        state = self.segmentState(segmentId)
        state.restore(stack.pop())
        if display:
            mask = state.full_mask if state.full_mask is not None else np.zeros(volumeShape(volumeNode), np.uint8)
            self.showSegment(volumeNode, segmentationNode, segmentId, mask, None)
            name = segmentationNode.GetSegmentation().GetSegment(segmentId).GetName()
            self.log(f"UNDO: segment={name} iteration={state.iteration}")
        return True

    def clearSegment(self, volumeNode, segmentationNode, segmentId):
        self.forgetSegment(segmentId)
        self.showSegment(volumeNode, segmentationNode, segmentId, np.zeros(volumeShape(volumeNode), np.uint8), None)

    # ---------- display ----------
    def showSegment(self, volumeNode, segmentationNode, segmentId, mask, previousMask=None):
        """Write ``mask`` into the segment, and the voxels added/removed since ``previousMask`` into Δ segments."""
        segmentation = segmentationNode.GetSegmentation()
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        slicer.util.updateSegmentBinaryLabelmapFromArray(mask, segmentationNode, segmentId, volumeNode)

        current = mask > 0
        previous = previousMask > 0 if previousMask is not None else current
        for name, delta in ((SEG_ADDED, current & ~previous), (SEG_REMOVED, previous & ~current)):
            deltaId = segmentation.GetSegmentIdBySegmentName(name)
            if not delta.any() and not deltaId:
                continue
            if not deltaId:
                deltaId = segmentation.AddEmptySegment("", name, DELTA_COLORS[name])
            slicer.util.updateSegmentBinaryLabelmapFromArray(delta.astype(np.uint8), segmentationNode, deltaId, volumeNode)

        segmentationNode.CreateDefaultDisplayNodes()
        displayNode = segmentationNode.GetDisplayNode()
        displayNode.SetPreferredDisplayRepresentationName2D("Binary labelmap")
        if not segmentation.ContainsRepresentation("Closed surface"):
            segmentationNode.CreateClosedSurfaceRepresentation()
            layoutManager = slicer.app.layoutManager()
            if layoutManager is not None:  # None when running without a main window
                if layoutManager.layout != slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView:
                    layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
                slicer.util.resetThreeDViews()

    def updateUncertaintyVolume(self, volumeNode, segmentId, uncertaintyNode=None):
        """Write the segment's critic uncertainty into a scalar volume (created if needed); None if not run."""
        uncertainty = InferenceEngine.uncertainty_map(volumeShape(volumeNode), self.segmentState(segmentId))
        if uncertainty is None:
            return None
        if uncertaintyNode is None:
            uncertaintyNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "SAT3D uncertainty")
            uncertaintyNode.SetHideFromEditors(True)
        uncertaintyNode.CopyOrientation(volumeNode)
        if uncertaintyNode.GetParentTransformNode() != volumeNode.GetParentTransformNode():
            uncertaintyNode.SetAndObserveTransformNodeID(volumeNode.GetTransformNodeID())
        slicer.util.updateVolumeFromArray(uncertaintyNode, np.nan_to_num(uncertainty, nan=-1.0).astype(np.float32))
        uncertaintyNode.CreateDefaultDisplayNodes()
        display = uncertaintyNode.GetDisplayNode()
        display.SetAndObserveColorNodeID(self._uncertaintyColorNode().GetID())
        display.AutoWindowLevelOff()
        display.SetWindowLevelMinMax(0.0, 1.0)
        display.SetApplyThreshold(True)  # unscored voxels (-1) become transparent
        display.SetThreshold(0.0, 1.0)
        display.SetInterpolate(False)
        return uncertaintyNode

    @staticmethod
    def _uncertaintyColorNode():
        """Blue (relatively certain) -> red (relatively likely mis-segmented)."""
        name = "SAT3D uncertainty colors"
        node = slicer.mrmlScene.GetFirstNodeByName(name)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLProceduralColorNode", name)
            node.SetHideFromEditors(True)
            node.SetAttribute("Category", "SAT3D")
            ctf = node.GetColorTransferFunction()
            ctf.AddRGBPoint(0.0, 50 / 255, 110 / 255, 230 / 255)
            ctf.AddRGBPoint(1.0, 230 / 255, 50 / 255, 50 / 255)
        return node

    # ---------- measurements / review ----------
    @staticmethod
    def resultSegmentIds(segmentationNode):
        """Segment IDs that hold results (excludes the Δ change overlays)."""
        segmentation = segmentationNode.GetSegmentation()
        return [s for s in segmentation.GetSegmentIDs() if segmentation.GetSegment(s).GetName() not in DELTA_SEGMENT_NAMES]

    def measurements(self, volumeNode, segmentationNode):
        """[(segmentId, name, SegmentMeasurements | None, approved)] for every result segment."""
        rows = []
        segmentation = segmentationNode.GetSegmentation()
        for segmentId in self.resultSegmentIds(segmentationNode):
            segment = segmentation.GetSegment(segmentId)
            mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, volumeNode)
            measured = measure_mask(mask, volumeNode.GetSpacing()) if mask is not None else None
            rows.append((segmentId, segment.GetName(), measured, self.isApproved(segment)))
        return rows

    @staticmethod
    def isApproved(segment):
        tag = vtk.reference("")
        return segment.GetTag(APPROVED_TAG, tag) and str(tag) == "1"

    def setApproved(self, segmentationNode, segmentId, approved=True):
        segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
        segment.SetTag(APPROVED_TAG, "1" if approved else "0")
        segmentationNode.Modified()
        self.log(f"{'APPROVED' if approved else 'UNAPPROVED'}: segment={segment.GetName()}")

    # ---------- saving ----------
    def _saveRun(self, volumeNode, segmentName, iteration, mask, prompts):
        os.makedirs(self.sessionDir, exist_ok=True)
        stem = f"{safeName(self.caseName)}_{safeName(segmentName)}_{iteration}_{datetime.now().strftime('%Y-%m-%d %H%M%S')}"
        saveMaskAsNifti(mask, volumeNode, os.path.join(self.sessionDir, f"{stem}.nii.gz"))
        self._writeProvenance(os.path.join(self.sessionDir, f"{stem}_prompts.json"), segmentName, prompts, None)

    def _writeProvenance(self, path, segmentName, prompts, segmentId):
        state = self.segmentStates.get(segmentId) if segmentId else None
        data = {
            "segment": segmentName,
            "points": [{"coord_dhw": list(c), "positive": p} for c, p in prompts.points],
            "box": {"min_corner_dhw": list(prompts.box[0]), "max_corner_dhw": list(prompts.box[1])} if prompts.box else None,
            "text": prompts.text,
            "roi_bounds": list(state.roi_bounds) if state and state.roi_bounds else None,
            "threshold": state.threshold if state else None,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def saveSegmentation(self, volumeNode, segmentationNode, outDir, promptsBySegment):
        """Multi-label NIfTI (result segment i -> label i) plus a prompt-provenance JSON per segment."""
        os.makedirs(outDir, exist_ok=True)
        segmentation = segmentationNode.GetSegmentation()
        combined = np.zeros(volumeShape(volumeNode), dtype=np.uint8)
        labels = {}
        for label, segmentId in enumerate(self.resultSegmentIds(segmentationNode), start=1):
            mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, volumeNode)
            if mask is not None:
                combined[mask > 0] = label
            segment = segmentation.GetSegment(segmentId)
            labels[label] = {"segment": segment.GetName(), "approved": self.isApproved(segment)}
            prompts = promptsBySegment.get(segmentId)
            if prompts is not None and (prompts.points or prompts.box or prompts.text):
                self._writeProvenance(os.path.join(outDir, f"{safeName(self.caseName)}_{safeName(segment.GetName())}_prompts.json"),
                                      segment.GetName(), prompts, segmentId)
        path = os.path.join(outDir, f"{safeName(self.caseName)}_seg.nii.gz")
        saveMaskAsNifti(combined, volumeNode, path)
        with open(os.path.join(outDir, f"{safeName(self.caseName)}_seg_labels.json"), "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
        self.log(f"SAVED: {path}")
        return path

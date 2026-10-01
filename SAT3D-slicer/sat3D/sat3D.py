import logging
import os
import time
from dataclasses import dataclass, field

import ctk
import numpy as np
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget
from slicer.util import VTKObservationMixin

from sat3DLib.engine import Prompts, sample_scribble
from sat3DLib.sat3DLogic import (
    BOX_REF,
    CURRENT_SEGMENT_PARAM,
    DEFAULT_SEGMENT_COLOR,
    DEFAULT_SEGMENT_NAME,
    DELTA_SEGMENT_NAMES,
    EXCLUDE_POINTS_REF,
    INCLUDE_POINTS_REF,
    INPUT_VOLUME_REF,
    SCRIBBLE_NEG,
    SCRIBBLE_POS,
    SCRIBBLE_SEGMENTATION_REF,
    SEGMENTATION_REF,
    UNCERTAINTY_VOLUME_REF,
    isVoxelInVolume,
    sat3DLogic,
    volumeShape,
    worldToVoxel,
)

logger = logging.getLogger("SAT3D")


class sat3D(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SAT3D"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Himashi Peiris"]
        self.parent.helpText = (
            "Interactive 3D tumour segmentation with SAT3D-plus.<br>"
            "Add include (<b>1</b>) / exclude (<b>2</b>) points, brush scribbles or a box (optionally with a text "
            "description) and click <b>Run / Refine</b>. Further prompts inside the same region refine the result "
            "quickly; prompts far away start a new 128³ crop.<br>"
            "Shortcuts: <b>a</b> clear prompts, <b>z</b> undo last run, <b>n</b> new segment.<br>"
            "Runs, prompts and a session log are written to <i>&lt;scan&gt;_sat3d_session</i> next to the scan. "
            "Model weights default to your Downloads folder (see Advanced).")


@dataclass
class PromptSnapshot:
    """Prompts of a segment that is not currently being edited."""
    include: list = field(default_factory=list)   # [(clickOrder, worldPos)]
    exclude: list = field(default_factory=list)
    scribblePos: np.ndarray = None
    scribbleNeg: np.ndarray = None
    box: tuple = None                              # (center, size, objectToNode as 16 floats)
    text: str = ""
    prompts: Prompts = None                        # voxel prompts at snapshot time (for provenance)


class sat3DWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self._observedMarkups = []
        self._observedSegmentation = None
        self._shortcuts = []
        self.frozenSliceView = None
        self._promptStore = {}        # segmentId -> PromptSnapshot
        self._activeSegmentId = None  # segment whose prompts are currently in the markups/scribble nodes
        self._clickOrder = {}         # (markupsNodeId, controlPointId) -> order
        self._nextClick = 0
        self._runTarget = None

    # -------------------- Setup --------------------
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/sat3D.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.logic = sat3DLogic()

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAddedEvent, self.onNodeAdded)

        ui = self.ui
        ui.comboVolumeNode.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        ui.segmentSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        ui.segmentSelector.connect("currentSegmentChanged(QString)", self.updateParameterNodeFromGUI)
        ui.markupsInclude.connect("markupsNodeChanged()", self.updateParameterNodeFromGUI)
        ui.markupsExclude.connect("markupsNodeChanged()", self.updateParameterNodeFromGUI)
        ui.markupsInclude.markupsPlaceWidget().setPlaceModePersistency(True)
        ui.markupsExclude.markupsPlaceWidget().setPlaceModePersistency(True)

        ui.pushPlaceBox.connect("clicked(bool)", self.onPlaceBox)
        ui.pushRemoveBox.connect("clicked(bool)", self.onRemoveBox)
        ui.pushBrushInclude.connect("clicked(bool)", lambda on: self.setBrush("include" if on else None))
        ui.pushBrushExclude.connect("clicked(bool)", lambda on: self.setBrush("exclude" if on else None))
        ui.pushBrushErase.connect("clicked(bool)", lambda on: self.setBrush("erase" if on else None))
        ui.pushClearPoints.connect("clicked(bool)", self.clearPrompts)

        ui.selfensembling.connect("clicked(bool)", self.onRunModel)
        ui.thresholdSlider.tracking = False
        ui.thresholdSlider.connect("valueChanged(double)", self.onThresholdChanged)
        ui.showUncertaintyCheckBox.connect("toggled(bool)", lambda on: self.updateUncertaintyOverlay())
        ui.pushUndo.connect("clicked(bool)", self.onUndo)
        ui.pushClearSegment.connect("clicked(bool)", self.onClearSegment)

        ui.pushSegmentAdd.connect("clicked(bool)", self.onSegmentAdd)
        ui.pushSegmentRemove.connect("clicked(bool)", self.onSegmentRemove)
        ui.pushApprove.connect("clicked(bool)", self.onApprove)
        ui.pushSave.connect("clicked(bool)", self.onSave)
        ui.endTask.connect("clicked(bool)", self.onEndTask)
        ui.measurementsTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeToContents)

        self._setupAdvanced()
        self._setupScribbleEditor()

        self._pollTimer = qt.QTimer()
        self._pollTimer.setInterval(20)  # each poll blocks ~80 ms so the worker gets the GIL
        self._pollTimer.connect("timeout()", self._onPollPrediction)
        self._measureTimer = qt.QTimer()
        self._measureTimer.setSingleShot(True)
        self._measureTimer.setInterval(500)
        self._measureTimer.connect("timeout()", self.updateMeasurements)

        for key, callback in (
            ("1", lambda: self.activatePlacement(INCLUDE_POINTS_REF)),
            ("2", lambda: self.activatePlacement(EXCLUDE_POINTS_REF)),
            ("a", self.clearPrompts),
            ("n", self.onSegmentAdd),
            ("z", self.onUndo),
        ):
            shortcut = qt.QShortcut(qt.QKeySequence(key), slicer.util.mainWindow())
            shortcut.connect("activated()", callback)
            shortcut.setEnabled(False)  # only active while the module is shown
            self._shortcuts.append(shortcut)

        self.initializeParameterNode()

    def _setupAdvanced(self):
        ui, s = self.ui, self.logic.settings
        for widget, name in ((ui.samWeightsPath, "samCheckpoint"), (ui.criticWeightsPath, "criticCheckpoint")):
            widget.filters = ctk.ctkPathLineEdit.Files | ctk.ctkPathLineEdit.Readable
            widget.nameFilters = ["PyTorch checkpoint (*.pth *.pt)"]
            widget.currentPath = getattr(s, name)
            widget.connect("currentPathChanged(QString)", lambda path, n=name: self._onModelSettingChanged(n, path))
        ui.textEncoderPath.filters = ctk.ctkPathLineEdit.Dirs
        ui.textEncoderPath.currentPath = s.textEncoderDir
        ui.textEncoderPath.connect("currentPathChanged(QString)", lambda path: self._onModelSettingChanged("textEncoderDir", path))
        ui.deviceCombo.currentText = s.device
        ui.deviceCombo.connect("currentTextChanged(QString)", lambda text: self._onModelSettingChanged("device", text))
        ui.randomSeed.value = s.seed
        ui.randomSeed.connect("valueChanged(int)", lambda v: self._onModelSettingChanged("seed", v))
        for widget, name in ((ui.roiMarginSpin, "roiMargin"), (ui.scribbleStrideSpin, "scribbleStride"),
                             (ui.maxScribbleSpin, "maxScribblePoints"), (ui.boxDepthSpin, "boxDepth")):
            widget.value = getattr(s, name)
            widget.connect("valueChanged(int)", lambda v, n=name: self._onSettingChanged(n, v))
        ui.autoSaveCheckBox.checked = s.autoSaveRuns
        ui.autoSaveCheckBox.connect("toggled(bool)", lambda on: s.set("autoSaveRuns", on))
        ui.pushReloadModel.connect("clicked(bool)", self.onUnloadModel)

    def _onSettingChanged(self, name, value):
        self.logic.settings.set(name, value)
        if name == "roiMargin" and self.logic.engine is not None:
            self.logic.engine.roi_margin = value

    def _onModelSettingChanged(self, name, value):
        self.logic.settings.set(name, value)
        if self.logic.modelLoaded and not self.logic.busy:
            self.onUnloadModel()
            self.setStatus("Settings changed: the model will reload on the next run.")

    def onUnloadModel(self):
        if self.logic.busy:
            return
        if self.logic.modelLoaded:
            self.logic.unloadModel()
        self.ui.modelStatusLabel.text = "not loaded"

    # -------------------- Lifecycle --------------------
    def cleanup(self):
        self._cleanedUp = True  # widgets may emit signals while being destroyed at shutdown
        self.removeObservers()
        self._pollTimer.stop()
        self.setBrush(None)
        self.unfreezeSlice()
        for shortcut in self._shortcuts:
            shortcut.setEnabled(False)
            shortcut.deleteLater()
        self._shortcuts = []

    def enter(self):
        self.initializeParameterNode()
        for shortcut in self._shortcuts:
            shortcut.setEnabled(True)

    def exit(self):
        for shortcut in self._shortcuts:
            shortcut.setEnabled(False)
        self.setBrush(None)
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        self.setBrush(None)
        self.unfreezeSlice()
        self.setParameterNode(None)
        self._resetPromptBookkeeping()
        self.logic.resetCase()

    def onSceneEndClose(self, caller, event):
        self._setupScribbleEditorNode()
        if self.parent.isEntered:
            self.initializeParameterNode()

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def onNodeAdded(self, caller, event, node):
        # Auto-select newly loaded images (exact class check skips label maps; our own outputs are hidden)
        if node.GetClassName() == "vtkMRMLScalarVolumeNode":
            def select():
                if node.GetScene() and not node.GetHideFromEditors():
                    self.ui.comboVolumeNode.setCurrentNode(node)
            qt.QTimer.singleShot(0, select)

    def _resetPromptBookkeeping(self):
        self._promptStore = {}
        self._activeSegmentId = None
        self._clickOrder = {}

    # -------------------- Parameter node --------------------
    def initializeParameterNode(self):
        selectedVolume = self.ui.comboVolumeNode.currentNode()  # read before the GUI is synced from the node
        self.setParameterNode(self.logic.getParameterNode())
        pn = self._parameterNode

        if not pn.GetNodeReference(INPUT_VOLUME_REF) and selectedVolume:
            pn.SetNodeReferenceID(INPUT_VOLUME_REF, selectedVolume.GetID())

        for ref, name, color in ((INCLUDE_POINTS_REF, "include-points", (0, 1, 0)),
                                 (EXCLUDE_POINTS_REF, "exclude-points", (1, 0, 0))):
            if not pn.GetNodeReference(ref):
                node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
                node.CreateDefaultDisplayNodes()
                displayNode = node.GetDisplayNode()
                displayNode.SetSelectedColor(*color)
                displayNode.SetActiveColor(*color)
                displayNode.SetTextScale(0)
                displayNode.SetGlyphScale(1)
                pn.SetNodeReferenceID(ref, node.GetID())

        if not pn.GetNodeReference(SEGMENTATION_REF):
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Segmentation")
            segmentationNode.CreateDefaultDisplayNodes()
            segmentId = segmentationNode.GetSegmentation().AddEmptySegment("", DEFAULT_SEGMENT_NAME, DEFAULT_SEGMENT_COLOR)
            pn.SetNodeReferenceID(SEGMENTATION_REF, segmentationNode.GetID())
            pn.SetParameter(CURRENT_SEGMENT_PARAM, segmentId)

        if not pn.GetNodeReference(SCRIBBLE_SEGMENTATION_REF):
            scribbles = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "SAT3D scribbles")
            scribbles.SetHideFromEditors(True)
            scribbles.CreateDefaultDisplayNodes()
            scribbles.GetDisplayNode().SetOpacity2DFill(0.6)
            scribbles.GetSegmentation().AddEmptySegment(SCRIBBLE_POS, SCRIBBLE_POS, (0.0, 1.0, 0.0))
            scribbles.GetSegmentation().AddEmptySegment(SCRIBBLE_NEG, SCRIBBLE_NEG, (1.0, 0.0, 0.0))
            pn.SetNodeReferenceID(SCRIBBLE_SEGMENTATION_REF, scribbles.GetID())

        volumeNode = pn.GetNodeReference(INPUT_VOLUME_REF)
        if volumeNode and self.logic.caseName != volumeNode.GetName():
            self.onInputVolumeChanged(volumeNode)
        self._syncScribbleEditor()
        self._activeSegmentId = pn.GetParameter(CURRENT_SEGMENT_PARAM) or None
        self.onSegmentSwitched()

    def setParameterNode(self, inputParameterNode):
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        self._updatingGUIFromParameterNode = True
        try:
            pn = self._parameterNode
            self.ui.comboVolumeNode.setCurrentNode(pn.GetNodeReference(INPUT_VOLUME_REF))
            self.ui.markupsInclude.setCurrentNode(pn.GetNodeReference(INCLUDE_POINTS_REF))
            self.ui.markupsExclude.setCurrentNode(pn.GetNodeReference(EXCLUDE_POINTS_REF))
            segmentationNode = pn.GetNodeReference(SEGMENTATION_REF)
            self.ui.segmentSelector.setCurrentNode(segmentationNode)
            self.ui.segmentsTable.setSegmentationNode(segmentationNode)
            if pn.GetParameter(CURRENT_SEGMENT_PARAM):
                self.ui.segmentSelector.setCurrentSegmentID(pn.GetParameter(CURRENT_SEGMENT_PARAM))
            self._observeMarkups([pn.GetNodeReference(INCLUDE_POINTS_REF), pn.GetNodeReference(EXCLUDE_POINTS_REF)])
            self._observeSegmentation(segmentationNode)
            self._updateButtonStates()
        finally:
            self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode or getattr(self, "_cleanedUp", False):
            return
        pn = self._parameterNode
        wasModified = pn.StartModify()
        try:
            volumeChanged = self.ui.comboVolumeNode.currentNodeID != pn.GetNodeReferenceID(INPUT_VOLUME_REF)
            pn.SetNodeReferenceID(INPUT_VOLUME_REF, self.ui.comboVolumeNode.currentNodeID)
            for ref, widget in ((INCLUDE_POINTS_REF, self.ui.markupsInclude), (EXCLUDE_POINTS_REF, self.ui.markupsExclude)):
                pn.SetNodeReferenceID(ref, widget.currentNode().GetID() if widget.currentNode() else None)
            pn.SetNodeReferenceID(SEGMENTATION_REF, self.ui.segmentSelector.currentNodeID())
            pn.SetParameter(CURRENT_SEGMENT_PARAM, self.ui.segmentSelector.currentSegmentID() or "")
        finally:
            pn.EndModify(wasModified)
        if volumeChanged and pn.GetNodeReference(INPUT_VOLUME_REF):
            self.onInputVolumeChanged(pn.GetNodeReference(INPUT_VOLUME_REF))
        if (pn.GetParameter(CURRENT_SEGMENT_PARAM) or None) != self._activeSegmentId:
            self.onSegmentSwitched()

    def onInputVolumeChanged(self, volumeNode):
        """New case: reset refinement state, clear prompts and show the volume."""
        self.setBrush(None)
        self._clearPromptNodes()
        self._resetPromptBookkeeping()
        self._activeSegmentId = self._parameterNode.GetParameter(CURRENT_SEGMENT_PARAM) or None
        self.logic.resetCase(volumeNode)
        for ref in (SEGMENTATION_REF, SCRIBBLE_SEGMENTATION_REF):
            node = self._parameterNode.GetNodeReference(ref)
            if node:
                node.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        self._syncScribbleEditor()
        slicer.util.setSliceViewerLayers(background=volumeNode, foreground=None, label=None, fit=True)
        self.setStatus(f"Loaded {volumeNode.GetName()} {volumeShape(volumeNode)}.")
        self.updateMeasurements()

    def _updateButtonStates(self):
        pn = self._parameterNode
        if pn is None:
            return
        ready = pn.GetNodeReference(INPUT_VOLUME_REF) is not None and pn.GetNodeReference(SEGMENTATION_REF) is not None
        busy = self.logic.busy
        self.ui.selfensembling.enabled = ready and not busy
        self.ui.selfensembling.text = "Running..." if busy else "Run / Refine"
        for w in (self.ui.pushUndo, self.ui.pushClearSegment, self.ui.thresholdSlider, self.ui.pushSegmentRemove):
            w.enabled = ready and not busy
        self.ui.modelStatusLabel.text = self.logic.deviceName()

    def setStatus(self, text):
        self.ui.statusLabel.text = text
        slicer.util.showStatusMessage(f"SAT3D: {text}", 5000)

    # -------------------- Segment switching (per-segment prompts) --------------------
    def _currentSegmentId(self):
        return self._parameterNode.GetParameter(CURRENT_SEGMENT_PARAM) or None if self._parameterNode else None

    def onSegmentSwitched(self):
        """Stash the prompts of the previously edited segment and restore those of the newly selected one."""
        newId = self._currentSegmentId()
        if newId != self._activeSegmentId:
            segmentationNode = self._parameterNode.GetNodeReference(SEGMENTATION_REF)
            oldExists = (self._activeSegmentId and segmentationNode
                         and segmentationNode.GetSegmentation().GetSegment(self._activeSegmentId) is not None)
            if oldExists:
                self._promptStore[self._activeSegmentId] = self._snapshotPrompts()
            self._clearPromptNodes()
            if newId in self._promptStore:
                self._restorePrompts(self._promptStore.pop(newId))
            self._activeSegmentId = newId

        segment = self._currentSegment()
        state = self.logic.segmentStates.get(newId) if newId else None
        wasBlocked = self.ui.thresholdSlider.blockSignals(True)
        self.ui.thresholdSlider.value = state.threshold if state else 0.5
        self.ui.thresholdSlider.blockSignals(wasBlocked)
        self.ui.pushApprove.checked = bool(segment and self.logic.isApproved(segment))
        self.updateUncertaintyOverlay()

    def _currentSegment(self):
        segmentationNode = self._parameterNode.GetNodeReference(SEGMENTATION_REF) if self._parameterNode else None
        segmentId = self._currentSegmentId()
        return segmentationNode.GetSegmentation().GetSegment(segmentId) if segmentationNode and segmentId else None

    def _snapshotPrompts(self):
        snap = PromptSnapshot(text=self.ui.textPrompt.text)
        for node, target in ((self._node(INCLUDE_POINTS_REF), snap.include), (self._node(EXCLUDE_POINTS_REF), snap.exclude)):
            for i in self._definedPointIndices(node):
                order = self._clickOrder.get((node.GetID(), node.GetNthControlPointID(i)), float("inf"))
                target.append((order, node.GetNthControlPointPositionWorld(i)))
        volumeNode = self._node(INPUT_VOLUME_REF)
        snap.scribblePos = self._scribbleArray(SCRIBBLE_POS, volumeNode)
        snap.scribbleNeg = self._scribbleArray(SCRIBBLE_NEG, volumeNode)
        box = self._node(BOX_REF)
        if box is not None and box.GetNumberOfDefinedControlPoints() > 0:
            matrix = box.GetObjectToNodeMatrix()
            snap.box = (box.GetCenter(), box.GetSize(), [matrix.GetElement(r, c) for r in range(4) for c in range(4)])
        snap.prompts = self.currentPrompts(volumeNode) if volumeNode else None
        return snap

    def _restorePrompts(self, snap):
        for node, points in ((self._node(INCLUDE_POINTS_REF), snap.include), (self._node(EXCLUDE_POINTS_REF), snap.exclude)):
            for order, pos in sorted(points, key=lambda p: p[0]):
                index = node.AddControlPointWorld(vtk.vtkVector3d(*pos))
                node.SetNthControlPointLocked(index, True)
                self._recordClick(node, index)
        volumeNode = self._node(INPUT_VOLUME_REF)
        scribbles = self._node(SCRIBBLE_SEGMENTATION_REF)
        for name, array in ((SCRIBBLE_POS, snap.scribblePos), (SCRIBBLE_NEG, snap.scribbleNeg)):
            if array is not None and array.any() and volumeNode and scribbles:
                slicer.util.updateSegmentBinaryLabelmapFromArray(array, scribbles, name, volumeNode)
        if snap.box is not None:
            box = self._boxNode(create=True)
            center, size, matrix = snap.box
            objectToNode = vtk.vtkMatrix4x4()
            objectToNode.DeepCopy(matrix)
            box.SetCenter(center)
            box.SetSize(size)
            box.GetObjectToNodeMatrix().DeepCopy(objectToNode)
            box.Modified()
        self.ui.textPrompt.text = snap.text
        if snap.include or snap.exclude:
            self.freezeSlice(self.sliceViewForPoint(snap.include[0][1] if snap.include else snap.exclude[0][1]))

    def _clearPromptNodes(self):
        for node in self._promptNodes():
            node.RemoveAllControlPoints()
        volumeNode = self._node(INPUT_VOLUME_REF)
        scribbles = self._node(SCRIBBLE_SEGMENTATION_REF)
        if scribbles and volumeNode and volumeNode.GetImageData():
            empty = np.zeros(volumeShape(volumeNode), np.uint8)
            for name in (SCRIBBLE_POS, SCRIBBLE_NEG):
                if scribbles.GetSegmentation().GetSegment(name):
                    slicer.util.updateSegmentBinaryLabelmapFromArray(empty, scribbles, name, volumeNode)
        box = self._node(BOX_REF)
        if box is not None:
            box.RemoveAllControlPoints()
        self.ui.textPrompt.text = ""
        self.unfreezeSlice()

    # -------------------- Prompts --------------------
    def _node(self, ref):
        return self._parameterNode.GetNodeReference(ref) if self._parameterNode else None

    def _promptNodes(self):
        return [n for n in (self._node(INCLUDE_POINTS_REF), self._node(EXCLUDE_POINTS_REF)) if n]

    @staticmethod
    def _definedPointIndices(node):
        return [i for i in range(node.GetNumberOfControlPoints())
                if node.GetNthControlPointPositionStatus(i) == node.PositionDefined]

    def _recordClick(self, node, index):
        self._clickOrder[(node.GetID(), node.GetNthControlPointID(index))] = self._nextClick
        self._nextClick += 1

    def _scribbleArray(self, name, volumeNode):
        scribbles = self._node(SCRIBBLE_SEGMENTATION_REF)
        if scribbles is None or volumeNode is None or scribbles.GetSegmentation().GetSegment(name) is None:
            return None
        array = slicer.util.arrayFromSegmentBinaryLabelmap(scribbles, name, volumeNode)
        return array if array is not None and array.any() else None

    def currentPrompts(self, volumeNode):
        """The current segment's prompts as voxel (d, h, w) coordinates, points in click order."""
        settings = self.logic.settings
        ordered = []
        for node, positive in ((self._node(INCLUDE_POINTS_REF), True), (self._node(EXCLUDE_POINTS_REF), False)):
            if node is None:
                continue
            for i in self._definedPointIndices(node):
                voxel = worldToVoxel(volumeNode, node.GetNthControlPointPositionWorld(i))
                if isVoxelInVolume(volumeNode, voxel):
                    order = self._clickOrder.get((node.GetID(), node.GetNthControlPointID(i)), float("inf"))
                    ordered.append((order, len(ordered), (voxel, positive)))
        points = [p for _, _, p in sorted(ordered)]
        for name, positive in ((SCRIBBLE_POS, True), (SCRIBBLE_NEG, False)):
            array = self._scribbleArray(name, volumeNode)
            if array is not None:
                points += [(c, positive) for c in sample_scribble(array, settings.scribbleStride, settings.maxScribblePoints)]
        return Prompts(points=points, box=self._boxVoxelBounds(volumeNode), text=self.ui.textPrompt.text.strip() or None)

    def _boxVoxelBounds(self, volumeNode):
        box = self._node(BOX_REF)
        if box is None or box.GetNumberOfDefinedControlPoints() == 0:
            return None
        rasBounds = [0.0] * 6
        box.GetRASBounds(rasBounds)
        corners = np.array([worldToVoxel(volumeNode, (x, y, z))
                            for x in rasBounds[0:2] for y in rasBounds[2:4] for z in rasBounds[4:6]])
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        shape = volumeShape(volumeNode)
        half = self.logic.settings.boxDepth // 2
        for axis in range(3):
            if hi[axis] - lo[axis] < 2:  # drawn flat in one view: give it the configured depth
                centre = (lo[axis] + hi[axis]) // 2
                lo[axis], hi[axis] = centre - half, centre + half
        lo = np.clip(lo, 0, np.array(shape) - 1)
        hi = np.clip(hi, 0, np.array(shape) - 1)
        return tuple(int(v) for v in lo), tuple(int(v) for v in hi)

    def _observeMarkups(self, nodes):
        nodes = [n for n in nodes if n is not None]
        if nodes == self._observedMarkups:
            return
        for node in self._observedMarkups:
            self.removeObserver(node, slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.onPointDefined)
            self.removeObserver(node, slicer.vtkMRMLMarkupsNode.PointRemovedEvent, self.onPointRemoved)
        for node in nodes:
            self.addObserver(node, slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.onPointDefined)
            self.addObserver(node, slicer.vtkMRMLMarkupsNode.PointRemovedEvent, self.onPointRemoved)
        self._observedMarkups = nodes

    @vtk.calldata_type(vtk.VTK_INT)
    def onPointDefined(self, markupsNode, event, pointIndex):
        volumeNode = self._node(INPUT_VOLUME_REF)
        worldPos = markupsNode.GetNthControlPointPositionWorld(pointIndex)
        if volumeNode is None or not isVoxelInVolume(volumeNode, worldToVoxel(volumeNode, worldPos)):
            slicer.util.showStatusMessage("SAT3D: point ignored, it is outside the input volume.", 3000)
            qt.QTimer.singleShot(0, lambda: markupsNode.RemoveNthControlPoint(pointIndex))
            return
        markupsNode.SetNthControlPointLocked(pointIndex, True)
        self._recordClick(markupsNode, pointIndex)
        if self.frozenSliceView is None:
            self.freezeSlice(self.sliceViewForPoint(worldPos))

    def onPointRemoved(self, caller, event):
        if not any(node.GetNumberOfDefinedControlPoints() for node in self._promptNodes()):
            self.unfreezeSlice()

    def clearPrompts(self):
        if self._parameterNode is None:
            return
        self.setBrush(None)
        self._clearPromptNodes()

    def activatePlacement(self, refName):
        nodeId = self._parameterNode.GetNodeReferenceID(refName) if self._parameterNode else None
        if not nodeId:
            return
        self.setBrush(None)
        slicer.app.applicationLogic().GetSelectionNode().SetActivePlaceNodeID(nodeId)
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetPlaceModePersistence(True)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

    # -------------------- Box --------------------
    def _boxNode(self, create=False):
        box = self._node(BOX_REF)
        if box is None and create:
            box = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "SAT3D box")
            box.CreateDefaultDisplayNodes()
            box.GetDisplayNode().SetSelectedColor(0.2, 0.6, 1.0)
            box.GetDisplayNode().SetFillOpacity(0.05)
            self._parameterNode.SetNodeReferenceID(BOX_REF, box.GetID())
        return box

    def onPlaceBox(self):
        if self._parameterNode is None:
            return
        self.setBrush(None)
        box = self._boxNode(create=True)
        box.RemoveAllControlPoints()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsROINode")
        selectionNode.SetActivePlaceNodeID(box.GetID())
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetPlaceModePersistence(False)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

    def onRemoveBox(self):
        box = self._node(BOX_REF)
        if box is not None:
            box.RemoveAllControlPoints()

    # -------------------- Scribbles (embedded Paint / Erase) --------------------
    def _setupScribbleEditor(self):
        editor = slicer.qMRMLSegmentEditorWidget()
        editor.setMRMLScene(slicer.mrmlScene)
        editor.setSegmentationNodeSelectorVisible(False)
        editor.setSourceVolumeNodeSelectorVisible(False)
        editor.setSwitchToSegmentationsButtonVisible(False)
        editor.setEffectNameOrder(["Paint", "Erase"])
        editor.unorderedEffectsVisible = False
        editor.setEffectColumnCount(2)
        self.ui.scribbleEditorPlaceholder.layout().addWidget(editor)
        self.scribbleEditor = editor
        self._setupScribbleEditorNode()

    def _setupScribbleEditorNode(self):
        node = slicer.mrmlScene.GetSingletonNode("SAT3D", "vtkMRMLSegmentEditorNode")
        if node is None:
            node = slicer.vtkMRMLSegmentEditorNode()
            node.SetSingletonTag("SAT3D")
            node.SetHideFromEditors(True)
            slicer.mrmlScene.AddNode(node)
        self.scribbleEditor.setMRMLSegmentEditorNode(node)

    def _syncScribbleEditor(self):
        self.scribbleEditor.setSegmentationNode(self._node(SCRIBBLE_SEGMENTATION_REF))
        self.scribbleEditor.setSourceVolumeNode(self._node(INPUT_VOLUME_REF))

    def setBrush(self, mode):
        """mode: 'include' | 'exclude' | 'erase' | None (stop painting)."""
        buttons = {"include": self.ui.pushBrushInclude, "exclude": self.ui.pushBrushExclude, "erase": self.ui.pushBrushErase}
        for key, button in buttons.items():
            button.checked = key == mode
        if mode is None:
            self.scribbleEditor.setActiveEffect(None)
            return
        if self._node(INPUT_VOLUME_REF) is None:
            buttons[mode].checked = False
            slicer.util.errorDisplay("Select an input volume first.")
            return
        self._syncScribbleEditor()
        if mode in ("include", "exclude"):
            self.scribbleEditor.setCurrentSegmentID(SCRIBBLE_POS if mode == "include" else SCRIBBLE_NEG)
        slicer.app.applicationLogic().GetInteractionNode().SwitchToViewTransformMode()
        self.scribbleEditor.setActiveEffectByName("Paint" if mode != "erase" else "Erase")

    # -------------------- Slice freezing --------------------
    @staticmethod
    def sliceViewForPoint(worldPos):
        """Name of the slice view the point was placed in: the view under the mouse, else the closest slice plane."""
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return None
        names = [n for n in layoutManager.sliceViewNames() if layoutManager.sliceWidget(n).sliceView().isVisible()]
        cursor = qt.QCursor.pos()
        for name in names:
            view = layoutManager.sliceWidget(name).sliceView()
            if view.rect.contains(view.mapFromGlobal(cursor)):
                return name

        def distanceToPlane(name):
            sliceToRas = layoutManager.sliceWidget(name).mrmlSliceNode().GetSliceToRAS()
            normal = [sliceToRas.GetElement(r, 2) for r in range(3)]
            origin = [sliceToRas.GetElement(r, 3) for r in range(3)]
            return abs(np.dot(np.subtract(worldPos, origin), normal))
        return min(names, key=distanceToPlane) if names else None

    def _setSliceFrozen(self, viewName, frozen):
        layoutManager = slicer.app.layoutManager()
        sliceWidget = layoutManager.sliceWidget(viewName) if viewName and layoutManager else None
        if sliceWidget is None:
            return
        interactorStyle = sliceWidget.sliceView().sliceViewInteractorStyle()
        interactorStyle.SetActionEnabled(interactorStyle.BrowseSlice, not frozen)
        sliceWidget.sliceView().setBackgroundColor(qt.QColor.fromRgbF(*((1, 1, 1) if frozen else (0, 0, 0))))
        sliceWidget.sliceController().setDisabled(frozen)

    def freezeSlice(self, viewName):
        """Lock slice browsing in the view where prompting started (white background marks it as frozen)."""
        self.unfreezeSlice()
        self._setSliceFrozen(viewName, True)
        self.frozenSliceView = viewName

    def unfreezeSlice(self):
        self._setSliceFrozen(self.frozenSliceView, False)
        self.frozenSliceView = None

    # -------------------- Run / refine --------------------
    def _currentTarget(self):
        """(volume, segmentation, segmentId) to run on, or None after telling the user what is missing."""
        volumeNode = self._node(INPUT_VOLUME_REF)
        segmentationNode = self._node(SEGMENTATION_REF)
        segment = self._currentSegment()
        if volumeNode is None or volumeNode.GetImageData() is None:
            slicer.util.errorDisplay("Select an input volume.")
            return None
        if segment is None:
            slicer.util.errorDisplay("Select a target segment.")
            return None
        if segment.GetName() in DELTA_SEGMENT_NAMES:
            slicer.util.errorDisplay(f'"{segment.GetName()}" only shows the change from the last run. Select another segment.')
            return None
        return volumeNode, segmentationNode, self._currentSegmentId()

    def ensureModelLoaded(self):
        if self.logic.modelLoaded:
            return True
        self.setStatus("Loading model (first time can take a while)...")
        slicer.app.processEvents()
        try:
            with slicer.util.tryWithErrorDisplay("Failed to load the SAT3D model.", waitCursor=True):
                loaded = self.logic.loadModel()
        except Exception:
            logger.exception("MODEL-LOAD-FAILED")
            self.setStatus("Model load failed.")
            return False
        self._updateButtonStates()
        return loaded

    def onRunModel(self):
        if self.logic.busy:
            return
        target = self._currentTarget()
        if target is None:
            return
        volumeNode, segmentationNode, segmentId = target
        prompts = self.currentPrompts(volumeNode)
        if not prompts.hasSpatial:
            slicer.util.infoDisplay("Add at least one point, scribble or box before running.\n"
                                    "Text alone can't localise a lesion.")
            return
        if not self.ensureModelLoaded():
            return
        try:
            self.logic.startPrediction(volumeNode, segmentId, prompts)
        except Exception as e:
            slicer.util.errorDisplay(f"SAT3D prediction failed.\n\n{e}")
            return
        self._runTarget = (volumeNode, segmentationNode)
        self._updateButtonStates()
        self._pollTimer.start()

    def _onPollPrediction(self):
        volumeNode, segmentationNode = self._runTarget
        job = self.logic.pollPrediction(volumeNode, segmentationNode)
        if job is None:
            elapsed = time.time() - self.logic._job["start"]
            self.setStatus(f"Running SAT3D on {self.logic.deviceName()}... {elapsed:.0f}s")
            return
        self._pollTimer.stop()
        self._runTarget = None
        self._updateButtonStates()
        if job["error"] is not None:
            self.setStatus(f"Run failed after {job['elapsed']:.1f}s.")
            slicer.util.errorDisplay(f"SAT3D prediction failed.\n\n{job['error']}", detailedText=job.get("traceback"))
            return
        message = f"Run complete in {job['elapsed']:.1f}s on {self.logic.deviceName()}."
        if job.get("tightened"):
            message += " The prompts spanned more than the model's 128-voxel window, so the crop was tightened."
        self.setStatus(message)
        self.updateUncertaintyOverlay()
        self.updateMeasurements()

    def onThresholdChanged(self, value):
        target = self._currentTarget() if not self.logic.busy else None
        if target and self.logic.rethreshold(*target, value):
            self.updateMeasurements()

    def onUndo(self):
        if self.logic.busy:
            return
        target = self._currentTarget()
        if target is None:
            return
        if self.logic.undo(*target):
            self.onSegmentSwitched()  # resync threshold / overlay with the restored state
            self.updateMeasurements()
        else:
            self.setStatus("Nothing to undo.")

    def onClearSegment(self):
        target = self._currentTarget() if not self.logic.busy else None
        if target is None:
            return
        self.clearPrompts()
        self.logic.clearSegment(*target)
        self.onSegmentSwitched()
        self.updateMeasurements()

    # -------------------- Uncertainty overlay --------------------
    def updateUncertaintyOverlay(self):
        if self._parameterNode is None:
            return
        node = self._node(UNCERTAINTY_VOLUME_REF)
        volumeNode = self._node(INPUT_VOLUME_REF)
        segmentId = self._currentSegmentId()
        if self.ui.showUncertaintyCheckBox.checked and volumeNode and segmentId:
            updated = self.logic.updateUncertaintyVolume(volumeNode, segmentId, node)
            if updated is not None:
                if node is None:
                    self._parameterNode.SetNodeReferenceID(UNCERTAINTY_VOLUME_REF, updated.GetID())
                slicer.util.setSliceViewerLayers(foreground=updated, foregroundOpacity=0.5)
                return
            if node is not None:
                self.setStatus("No uncertainty yet: run the model on this segment first.")
        if node is not None:
            for sliceCompositeNode in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
                if sliceCompositeNode.GetForegroundVolumeID() == node.GetID():
                    sliceCompositeNode.SetForegroundVolumeID(None)

    # -------------------- Segments, review, measurements --------------------
    def _observeSegmentation(self, segmentationNode):
        if segmentationNode is self._observedSegmentation:
            return
        events = [slicer.vtkSegmentation.SegmentAdded, slicer.vtkSegmentation.SegmentRemoved,
                  slicer.vtkSegmentation.SegmentModified,
                  getattr(slicer.vtkSegmentation, "SourceRepresentationModified", None)
                  or slicer.vtkSegmentation.MasterRepresentationModified]
        if self._observedSegmentation is not None:
            for event in events:
                self.removeObserver(self._observedSegmentation, event, self._scheduleMeasurements)
        if segmentationNode is not None:
            for event in events:
                self.addObserver(segmentationNode, event, self._scheduleMeasurements)
        self._observedSegmentation = segmentationNode

    def _scheduleMeasurements(self, caller=None, event=None):
        self._measureTimer.start()

    def updateMeasurements(self):
        table = self.ui.measurementsTable
        volumeNode, segmentationNode = self._node(INPUT_VOLUME_REF), self._node(SEGMENTATION_REF)
        rows = self.logic.measurements(volumeNode, segmentationNode) if volumeNode and segmentationNode else []
        table.setRowCount(len(rows))
        for row, (_, name, measured, approved) in enumerate(rows):
            values = (name, f"{measured.volume_cm3:.2f}" if measured else "–",
                      f"{measured.longest_diameter_mm:.1f}" if measured else "–", "✓" if approved else "")
            for column, value in enumerate(values):
                table.setItem(row, column, qt.QTableWidgetItem(value))
        segment = self._currentSegment()
        self.ui.pushApprove.checked = bool(segment and self.logic.isApproved(segment))

    def onApprove(self, approved):
        segmentationNode, segmentId = self._node(SEGMENTATION_REF), self._currentSegmentId()
        if segmentationNode is None or segmentId is None:
            return
        self.logic.setApproved(segmentationNode, segmentId, approved)
        self.updateMeasurements()

    def onSegmentAdd(self):
        segmentationNode = self._node(SEGMENTATION_REF)
        if segmentationNode is None:
            return
        self.setBrush(None)
        segmentId = segmentationNode.GetSegmentation().AddEmptySegment()
        self.ui.segmentSelector.setCurrentSegmentID(segmentId)

    def onSegmentRemove(self):
        segmentationNode, segmentId = self._node(SEGMENTATION_REF), self._currentSegmentId()
        if segmentationNode is None or not segmentId or self.logic.busy:
            return
        remaining = [s for s in self.logic.resultSegmentIds(segmentationNode) if s != segmentId]
        if not remaining:
            slicer.util.errorDisplay("Need to have at least one segment.")
            return
        self.logic.forgetSegment(segmentId)
        self._promptStore.pop(segmentId, None)
        self._clearPromptNodes()
        self._activeSegmentId = None
        segmentationNode.RemoveSegment(segmentId)
        self.ui.segmentSelector.setCurrentSegmentID(remaining[-1])

    def onSave(self):
        volumeNode, segmentationNode = self._node(INPUT_VOLUME_REF), self._node(SEGMENTATION_REF)
        if volumeNode is None or segmentationNode is None:
            slicer.util.errorDisplay("Select an input volume first.")
            return
        outDir = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), "Choose output folder",
                                                     self.logic.sessionDir or "")
        if not outDir:
            return
        promptsBySegment = {segmentId: snap.prompts for segmentId, snap in self._promptStore.items()}
        if self._currentSegmentId():
            promptsBySegment[self._currentSegmentId()] = self.currentPrompts(volumeNode)
        try:
            with slicer.util.tryWithErrorDisplay("Saving failed.", waitCursor=True):
                path = self.logic.saveSegmentation(volumeNode, segmentationNode, outDir, promptsBySegment)
        except Exception:
            logger.exception("SAVE-FAILED")
            return
        self.setStatus(f"Saved {path}")

    def onEndTask(self):
        if not slicer.util.confirmOkCancelDisplay("End the segmentation task?\nSlicer will restart."):
            return
        self.logic.log(f"END-TASK: case={self.logic.caseName}")
        slicer.app.restart()

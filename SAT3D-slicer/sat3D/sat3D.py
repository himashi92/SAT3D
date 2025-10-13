import logging
import os
from datetime import datetime

import numpy as np
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

# Import your logic (updated version below)
from sat3DLib.sat3DLogic import sat3DLogic


class sat3D(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SAT3D"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = [""]


class sat3DWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)

        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

        self.slice_visible = True
        self.actual_remove_click = True
        self.mask_accepted = False
        self.slice_frozen = False
        self.first_freeze = False

        self.lm = slicer.app.layoutManager()
        self.segmentEditorWidget = None
        self.layout_id = 20000
        self.orientation = 'horizontal'
        self.view = "Red"

        self.download_location = qt.QStandardPaths.writableLocation(qt.QStandardPaths.DownloadLocation)
        self.SAT3D_weights_path = os.path.join(self.download_location, "sam_model_dice_best.pth")
        self.SAT3D_critic_weights_path = os.path.join(self.download_location, "critic_dice_best.pth")

        self.layouts = {}
        self.createLayouts()

        self.rand_seed = 2025  # default

        # simplified (no participant/modality)
        self.log_file_name = None
        self.current_task_name = None
        self.run_ai_count = 0

    # -------------------- Setup --------------------
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Load UI
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/sat3D.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Logger + logic
        self.setup_logger()
        self.logic = sat3DLogic()

        # Hide legacy UI controls if present
        for maybe in ("participantCombo", "channelCombo", "modalityCombo"):
            w = getattr(self.ui, maybe, None)
            if w is not None:
                w.hide()
                w.setEnabled(False)

        # Observers
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAddedEvent, self.onNodeAdded)

        # Wire UI
        self.ui.comboVolumeNode.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.segmentSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.segmentSelector.connect("currentSegmentChanged(QString)", self.updateParameterNodeFromGUI)
        self.ui.segmentsTable.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        self.ui.markupsInclude.connect("markupsNodeChanged()", self.updateParameterNodeFromGUI)
        self.ui.markupsExclude.connect("markupsNodeChanged()", self.updateParameterNodeFromGUI)
        self.ui.markupsInclude.markupsPlaceWidget().setPlaceModePersistency(True)
        self.ui.markupsExclude.markupsPlaceWidget().setPlaceModePersistency(True)

        self.ui.pushSegmentAdd.connect("clicked(bool)", self.onPushSegmentAdd)
        self.ui.pushSegmentRemove.connect("clicked(bool)", self.onPushSegmentRemove)

        # "Run" + "End" buttons
        self.ui.selfensembling.connect("clicked(bool)", self.runAIModel)
        self.ui.endTask.connect("clicked(bool)", self.endTask)

        # Seed
        if hasattr(self.ui, "randomSeed"):
            self.ui.randomSeed.textChanged.connect(self.update_seed_value)

        # Shortcuts
        shortcuts = [
            ("1", self.activateIncludePoints),
            ("2", self.activateExcludePoints),
            ("a", self.onPushMaskAccept),
            ("n", self.onPushSegmentAdd),
            ("z", self.onPushUndo),
        ]
        for key, cb in shortcuts:
            sc = qt.QShortcut(qt.QKeySequence(key), slicer.util.mainWindow())
            sc.connect("activated()", cb)

        self.initializeParameterNode()
        self.logic.ui = self.ui  # attach UI to logic if needed

    # -------------------- Logging --------------------
    def setup_logger(self):
        self.log_file_name = f"{datetime.now().strftime('%Y-%m-%d %H%M%S')}_log.txt"
        self.logger = logging.getLogger("SlicerLogger")
        self.logger.setLevel(logging.INFO)
        # Avoid duplicate handlers
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            log_path = os.path.join(self.download_location, self.log_file_name)
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(fh)
        return self.logger

    def update_seed_value(self, text):
        try:
            self.rand_seed = int(text)
        except Exception:
            pass

    # -------------------- Display helpers --------------------
    def _fitAllSlices(self):
        lm = slicer.app.layoutManager()
        for viewName in ("Red", "Green", "Yellow"):
            sw = lm.sliceWidget(viewName)
            if not sw:
                continue
            # Prefer the controller helper if present; otherwise fall back to logic
            try:
                sw.sliceController().fitSliceToBackground()
            except Exception:
                try:
                    sw.sliceLogic().FitSliceToAll()
                except Exception:
                    pass

    def updateDisplayFromParameterNode(self):
        vol = self._parameterNode.GetNodeReference("fastsamInputVolume") if self._parameterNode else None
        if not vol or vol.GetImageData() is None:
            return
        try:
            _ = slicer.util.arrayFromVolume(vol)
        except Exception as e:
            self.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] updateDisplayFromParameterNode: Error: {e}")
            return
        self.displayChannel(vol)

    def displayChannel(self, volume_node):
        if volume_node is None or volume_node.GetImageData() is None:
            return
        try:
            _ = slicer.util.arrayFromVolume(volume_node)  # assert data exists
        except Exception as e:
            self.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR-DISPLAY: displayChannel: {e}")
            return

        # Use the original node as background (no resetFOV kwarg here)
        slicer.util.setSliceViewerLayers(
            background=volume_node, foreground=None, label=None
        )

        # Auto window/level on the original node’s display
        disp = volume_node.GetDisplayNode()
        if disp:
            disp.SetAutoWindowLevel(False)
            disp.SetAutoWindowLevel(True)

        # Fit all slice viewers explicitly
        self._fitAllSlices()

        slicer.app.processEvents()
        self.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] LOADING-DATASET: Slice viewers updated.")

    # -------------------- Parameter Node --------------------
    def initializeParameterNode(self):
        self.setParameterNode(self.logic.getParameterNode())
        self._parameterNode.root_path = os.path.dirname(os.path.dirname(os.path.dirname(self.resourcePath(''))))
        self._parameterNode.logger = self.logger

        # Use the single selected volume as input
        vol = self.ui.comboVolumeNode.currentNode()
        if vol is not None:
            self._parameterNode.SetNodeReferenceID("fastsamInputVolume", vol.GetID())
            self.current_task_name = vol.GetName()
            self.logic.set_current_case_name(self.current_task_name)
            self.displayChannel(vol)

        # Include points node
        if not self._parameterNode.GetNodeReferenceID("fastsamIncludePoints"):
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "include-points")
            self._parameterNode.SetNodeReferenceID("fastsamIncludePoints", node.GetID())
            dn = node.GetDisplayNode()
            dn.SetSelectedColor(0, 1, 0); dn.SetActiveColor(0, 1, 0)
            dn.SetTextScale(0); dn.SetGlyphScale(1)
            node.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.onMarkupIncludePointDefined)
            node.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionUndefinedEvent, self.onMarkupIncludePointUndefined)

        # Exclude points node
        if not self._parameterNode.GetNodeReferenceID("fastsamExcludePoints"):
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "exclude-points")
            self._parameterNode.SetNodeReferenceID("fastsamExcludePoints", node.GetID())
            dn = node.GetDisplayNode()
            dn.SetSelectedColor(1, 0, 0); dn.SetActiveColor(1, 0, 0)
            dn.SetTextScale(0); dn.SetGlyphScale(1)
            node.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.onMarkupExcludePointDefined)
            node.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionUndefinedEvent, self.onMarkupExcludePointUndefined)

        # Segmentation
        if not self._parameterNode.GetNodeReferenceID("fastsamSegmentation"):
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', 'Segmentation')
            self.ui.segmentSelector.setCurrentNode(segmentationNode)
            self.ui.segmentsTable.setSegmentationNode(segmentationNode)
            self._parameterNode.SetNodeReferenceID("fastsamSegmentation", segmentationNode.GetID())
            segmentationNode.CreateDefaultDisplayNodes()
            segmentID = segmentationNode.GetSegmentation().AddEmptySegment()
            self._parameterNode.SetParameter("fastsamCurrentSegment", segmentID)

        # Segment editor widget (for interpolation feature)
        if self.segmentEditorWidget is None:
            self.segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
            self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
            self.segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
            slicer.mrmlScene.AddNode(self.segmentEditorNode)
            self.segmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)
            self.segmentEditorWidget.setSegmentationNode(self._parameterNode.GetNodeReference("fastsamSegmentation"))
            self.segmentEditorWidget.setSourceVolumeNode(self._parameterNode.GetNodeReference("fastsamInputVolume"))
            self.segmentEditorWidget.setActiveEffectByName("Fill between slices")
            self.effect = self.segmentEditorWidget.activeEffect()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        self._updatingGUIFromParameterNode = True

        self.ui.comboVolumeNode.setCurrentNode(self._parameterNode.GetNodeReference("fastsamInputVolume"))
        self.ui.markupsInclude.setCurrentNode(self._parameterNode.GetNodeReference("fastsamIncludePoints"))
        self.ui.markupsExclude.setCurrentNode(self._parameterNode.GetNodeReference("fastsamExcludePoints"))

        if self._parameterNode.GetNodeReferenceID("fastsamSegmentation"):
            self._parameterNode.GetNodeReference("fastsamSegmentation").SetReferenceImageGeometryParameterFromVolumeNode(
                self._parameterNode.GetNodeReference("fastsamInputVolume"))
            self.ui.segmentsTable.setSegmentationNode(self._parameterNode.GetNodeReference("fastsamSegmentation"))
        if self.segmentEditorWidget:
            self.segmentEditorWidget.setSegmentationNode(self._parameterNode.GetNodeReference("fastsamSegmentation"))
            self.segmentEditorWidget.setSourceVolumeNode(self._parameterNode.GetNodeReference("fastsamInputVolume"))

        self.updateDisplayFromParameterNode()
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        wasModified = self._parameterNode.StartModify()

        if self.ui.comboVolumeNode.currentNodeID != self._parameterNode.GetNodeReferenceID("fastsamInputVolume"):
            self._parameterNode.SetNodeReferenceID("fastsamInputVolume", self.ui.comboVolumeNode.currentNodeID)
            vol = self._parameterNode.GetNodeReference("fastsamInputVolume")
            if vol:
                self.current_task_name = vol.GetName()
                self.logic.set_current_case_name(self.current_task_name)
            self.displayChannel(vol)

        if self._parameterNode.GetNodeReference("fastsamInputVolume") is not None:
            self.updateDisplayFromParameterNode()
            self.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INPUT-VOLUME-LOADED.")

        self._parameterNode.SetNodeReferenceID("fastsamIncludePoints", self.ui.markupsInclude.currentNode().GetID())
        self._parameterNode.SetNodeReferenceID("fastsamExcludePoints", self.ui.markupsExclude.currentNode().GetID())

        self._parameterNode.SetNodeReferenceID("fastsamSegmentation", self.ui.segmentSelector.currentNodeID())
        self._parameterNode.GetNodeReference("fastsamSegmentation").SetReferenceImageGeometryParameterFromVolumeNode(
            self._parameterNode.GetNodeReference("fastsamInputVolume"))
        self._parameterNode.SetParameter("fastsamCurrentSegment", self.ui.segmentSelector.currentSegmentID())
        self.ui.segmentsTable.setSegmentationNode(self._parameterNode.GetNodeReference("fastsamSegmentation"))
        if self.segmentEditorWidget:
            self.segmentEditorWidget.setSegmentationNode(self._parameterNode.GetNodeReference("fastsamSegmentation"))
            self.segmentEditorWidget.setSourceVolumeNode(self._parameterNode.GetNodeReference("fastsamInputVolume"))

        self._parameterNode.EndModify(wasModified)

    # -------------------- Markup helpers --------------------
    def findClickedSliceWindow(self):
        """
        Return the active slice view tag ('Red' | 'Green' | 'Yellow').
        Handles Slicer/Qt variants where isActiveWindow is a bool property
        or a callable. Falls back to 'Red'.
        """
        lm = self.lm
        for tag in ("Red", "Green", "Yellow"):
            sw = lm.sliceWidget(tag)
            if not sw:
                continue
            try:
                view = sw.sliceView()

                # Some builds expose isActiveWindow as a callable, others as a bool property.
                active_attr = getattr(view, "isActiveWindow", None)

                # Callable case
                if callable(active_attr):
                    if active_attr():
                        return tag

                # Bool-property case
                elif isinstance(active_attr, bool):
                    if active_attr:
                        return tag

                # Fallback: test via top-level window if available
                win = getattr(view, "window", None)
                if callable(win):
                    w = win()
                    if w is not None:
                        active_win_attr = getattr(w, "isActiveWindow", None)
                        if callable(active_win_attr):
                            if active_win_attr():
                                return tag
                        elif isinstance(active_win_attr, bool) and active_win_attr:
                            return tag

            except Exception:
                # Ignore and keep scanning other slice widgets
                pass

        # If we couldn't detect an active one, default to Red
        return "Red"

    def onMarkupIncludePointDefined(self, caller, event):
        slice_dir = self.findClickedSliceWindow()
        if caller.GetNumberOfControlPoints() == 1 and not self.slice_frozen:
            self.logic.slice_direction = slice_dir
            self.freezeSlice()
            self.first_freeze = True
        self.addPoint(caller, self.logic.include_coords)
        self.first_freeze = False

    def onMarkupExcludePointDefined(self, caller, event):
        slice_dir = self.findClickedSliceWindow()
        if caller.GetNumberOfControlPoints() == 1 and not self.slice_frozen:
            self.logic.slice_direction = slice_dir
            self.freezeSlice()
            self.first_freeze = True
        self.addPoint(caller, self.logic.exclude_coords)
        self.first_freeze = False

    def onMarkupIncludePointUndefined(self, caller, event):
        if caller.GetNumberOfControlPoints() == 0:
            self.unfreezeSlice()
            self.actual_remove_click = False
            caller.RemoveAllControlPoints()
            self.actual_remove_click = True
            self.logic.include_coords = {}
        if self.mask_accepted:
            self.mask_accepted = False
            return
        self.removePoint(caller, self.logic.include_coords, 'include')

    def onMarkupExcludePointUndefined(self, caller, event):
        if caller.GetNumberOfControlPoints() == 0:
            self.unfreezeSlice()
            self.actual_remove_click = False
            caller.RemoveAllControlPoints()
            self.actual_remove_click = True
            self.logic.exclude_coords = {}
        if self.mask_accepted:
            self.mask_accepted = False
            return
        self.removePoint(caller, self.logic.exclude_coords, 'exclude')

    # -------------------- Actions --------------------
    def runAIModel(self):
        self.run_ai_count += 1
        self.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] RUN-AI: Prediction Started")

        # lazy load weights
        if self.logic.sam is None:
            if not os.path.exists(self.SAT3D_weights_path) or not os.path.exists(self.SAT3D_critic_weights_path):
                slicer.util.errorDisplay(
                    f"SAT3D weights not found.\nExpected:\n{self.SAT3D_weights_path}\n{self.SAT3D_critic_weights_path}"
                )
                return
            self.logic.create_sam(self.SAT3D_weights_path, self.SAT3D_critic_weights_path,
                                  "swin2", self.rand_seed, self.log_file_name)

        self.logic.get_mask()

    def endTask(self):
        self.logic.endRefinementTask()

    def addPoint(self, caller, stored_coords):
        self.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ADDING_POINT")
        self.logic.generateaffine()
        point_index = caller.GetDisplayNode().GetActiveControlPoint()
        vol_node = self._parameterNode.GetNodeReference("fastsamInputVolume")
        if not vol_node:
            self.actual_remove_click = False
            caller.RemoveNthControlPoint(point_index)
            return

        # Prefer world (RAS) coords if available; fall back to Position (often RAS too)
        if hasattr(caller, "GetNthControlPointPositionWorld"):
            pos = caller.GetNthControlPointPositionWorld(point_index)  # tuple (x,y,z)
        else:
            pos = caller.GetNthControlPointPosition(point_index)  # tuple (x,y,z)

        # Build homogeneous RAS vector (tuple is fine; no .tolist())
        coords_ras = np.array([pos[0], pos[1], pos[2], 1.0], dtype=np.float32)

        # Map RAS -> IJK using cached inverse
        ras_to_ijk = getattr(self.logic, "ras_to_ijk", None)
        if ras_to_ijk is None:
            # should not happen because generateaffine() sets it
            self.logic.generateaffine()
            ras_to_ijk = self.logic.ras_to_ijk

        coords_ijk_h = ras_to_ijk @ coords_ras
        coords_ijk = np.rint(coords_ijk_h[:3]).astype(np.int32)

        # Lock the markup handle
        caller.SetNthControlPointLocked(point_index, True)

        # Reorder to (D,H,W) indexing for arrayFromVolume
        coords_dhw = np.array([coords_ijk[2], coords_ijk[1], coords_ijk[0]], dtype=np.int32)

        # Bounds check against actual image array
        img = slicer.util.arrayFromVolume(vol_node)  # (D,H,W)
        if not (0 <= coords_dhw[0] < img.shape[0] and 0 <= coords_dhw[1] < img.shape[1] and 0 <= coords_dhw[2] <
                img.shape[2]):
            self.actual_remove_click = False
            caller.RemoveNthControlPoint(point_index)
            return

        stored_coords[caller.GetNthControlPointLabel(point_index)] = coords_dhw
        self.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ADDED_POINT")

    def removePoint(self, caller, stored_coords, operation):
        if not self.actual_remove_click:
            self.actual_remove_click = True
            return
        new_coords = {}
        for i in range(caller.GetNumberOfControlPoints()):
            point_label = caller.GetNthControlPointLabel(i)
            if point_label in stored_coords:
                new_coords[point_label] = stored_coords[point_label]
        if operation == 'include':
            self.logic.include_coords = new_coords
        else:
            self.logic.exclude_coords = new_coords

    def clearPrevPromptPointsUI(self):
        inc_node = self._parameterNode.GetNodeReference("fastsamIncludePoints")
        exc_node = self._parameterNode.GetNodeReference("fastsamExcludePoints")
        self.actual_remove_click = False
        if inc_node and getattr(self.logic, "include_coords_prev", None):
            inc_labels_to_remove = set(self.logic.include_coords_prev.keys())
            for i in reversed(range(inc_node.GetNumberOfControlPoints())):
                lbl = inc_node.GetNthControlPointLabel(i)
                if lbl in inc_labels_to_remove:
                    inc_node.RemoveNthControlPoint(i)
            self.logic.include_coords_prev = {}
            self.removePoint(inc_node, self.logic.include_coords, 'include')
        if exc_node and getattr(self.logic, "exclude_coords_prev", None):
            exc_labels_to_remove = set(self.logic.exclude_coords_prev.keys())
            for i in reversed(range(exc_node.GetNumberOfControlPoints())):
                lbl = exc_node.GetNthControlPointLabel(i)
                if lbl in exc_labels_to_remove:
                    exc_node.RemoveNthControlPoint(i)
            self.logic.exclude_coords_prev = {}
            self.removePoint(exc_node, self.logic.exclude_coords, 'exclude')
        self.actual_remove_click = True

    def onPushMaskAccept(self):
        self.mask_accepted = True
        self.actual_remove_click = False
        node = self._parameterNode.GetNodeReference("fastsamIncludePoints")
        if node:
            node.RemoveAllControlPoints()
        self.logic.include_coords = {}

    def onPushMaskClear(self):
        if self.slice_frozen:
            self.logic.undo()
            self.onPushMaskAccept()

    def onPushUndo(self):
        self.logic.undo()

    def onPushSegmentAdd(self):
        self.clearPoints()
        self.logic.mask_locations = set()
        self.logic.interp_slice_direction = set()
        segNode = self._parameterNode.GetNodeReference("fastsamSegmentation")
        segmentID = segNode.GetSegmentation().AddEmptySegment()
        self._parameterNode.SetParameter("fastsamCurrentSegment", segmentID)
        self.ui.segmentSelector.setCurrentSegmentID(segmentID)
        # clear current working mask
        if isinstance(self.logic.mask, np.ndarray):
            self.logic.mask.fill(0)

    def onPushSegmentRemove(self):
        segNode = self._parameterNode.GetNodeReference("fastsamSegmentation")
        if len(segNode.GetSegmentation().GetSegmentIDs()) <= 1:
            slicer.util.errorDisplay("Need to have at least one segment")
            return
        segNode.RemoveSegment(self._parameterNode.GetParameter("fastsamCurrentSegment"))
        self.ui.segmentSelector.setCurrentSegmentID(segNode.GetSegmentation().GetSegmentIDs()[-1])

    def setlogitsview(self):
        self.logic.view = "logits"

    def logitsmaskdisplay(self):
        self.logic.pass_mask_to_slicer()

    def binarymaskdisplay(self):
        self.logic.pass_mask_to_slicer()

    def checkSAM(self):
        if self.logic.sam is None:
            slicer.util.errorDisplay("SAM weights not found, use Download button")
            return False
        return True

    def checkVolume(self):
        if not self._parameterNode.GetNodeReferenceID("fastsamInputVolume"):
            slicer.util.errorDisplay("Select a volume")
            return False
        else:
            return True

    def onPushInitializeInterp(self):
        # Uses Slicer "Fill between slices" effect
        if len(self.logic.interp_slice_direction) > 1:
            slicer.util.errorDisplay("Cannot interpolate if multiple slice directions have been segmented")
            return
        elif len(self.logic.interp_slice_direction) == 0 or len(self.logic.mask_locations) == 0:
            slicer.util.errorDisplay("Cannot interpolate if no masks have been added")
            return
        self.logic.backup_mask()
        segNode = self._parameterNode.GetNodeReference("fastsamSegmentation")
        for segmentID in segNode.GetSegmentation().GetSegmentIDs():
            if segmentID != self._parameterNode.GetParameter("fastsamCurrentSegment"):
                segNode.GetDisplayNode().SetSegmentVisibility(segmentID, False)
        self.effect.self().onPreview()
        self.effect.self().onApply()
        for segmentID in segNode.GetSegmentation().GetSegmentIDs():
            if segmentID != self._parameterNode.GetParameter("fastsamCurrentSegment"):
                segNode.GetDisplayNode().SetSegmentVisibility(segmentID, True)

    # -------------------- Scene + node wiring --------------------
    @vtk.calldata_type(vtk.VTK_OBJECT)
    def onNodeAdded(self, caller, event, calldata):
        node = calldata
        if type(node) == slicer.vtkMRMLScalarVolumeNode or node.IsA("vtkMRMLMultiVolumeNode") or node.IsA("vtkMRMLSequenceNode"):
            qt.QTimer.singleShot(30, lambda: self.autoSelectVolume(node))
            qt.QTimer.singleShot(30, lambda: self.importPKL(node))

    def autoSelectVolume(self, volumeNode):
        self.ui.comboVolumeNode.setCurrentNodeID(volumeNode.GetID())
        self.displayChannel(volumeNode)
        self._fitAllSlices()

    def importPKL(self, volumeNode):
        if volumeNode is not None:
            storage_node = volumeNode.GetStorageNode()
            if storage_node is not None:
                pkl_filepath = os.path.splitext(storage_node.GetFileName())[0] + ".pkl"
                if os.path.exists(pkl_filepath):
                    pl = getattr(self.ui, "PathLineEdit_emb", None)
                    if pl is not None:
                        pl.setCurrentPath(pkl_filepath)

    def activateIncludePoints(self):
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetActivePlaceNodeID(self._parameterNode.GetNodeReferenceID("fastsamIncludePoints"))
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

    def activateExcludePoints(self):
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetActivePlaceNodeID(self._parameterNode.GetNodeReferenceID("fastsamExcludePoints"))
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

    def createLayouts(self):
        orientations = ['horizontal', 'vertical']
        views = ['Red', 'Green', 'Yellow']
        for orientation in orientations:
            for view in views:
                self.layout_id += 1
                customLayout = f"""
                    <layout type="{orientation}" split="true">
                        <item>
                            <view class="vtkMRMLViewNode" singletontag="1">
                                <property name="viewlabel" action="default">1</property>
                            </view>
                        </item>
                        <item>
                            <view class="vtkMRMLSliceNode" singletontag="{view}">
                            </view>
                        </item>
                    </layout>
                """
                self.lm.layoutLogic().GetLayoutNode().AddLayoutDescription(self.layout_id, customLayout)
                self.layout_id += 1

    # -------------------- Lifecycle --------------------
    def cleanup(self):
        self.removeObservers()

    def enter(self):
        self.initializeParameterNode()

    def exit(self):
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        if self.parent.isEntered:
            self.initializeParameterNode()

    def setParameterNode(self, inputParameterNode):
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self.updateGUIFromParameterNode()

    # -------------------- Freeze / Unfreeze --------------------
    def freezeSlice(self):
        slice_widget = self.lm.sliceWidget(self.logic.slice_direction)
        if not slice_widget:
            return
        interactorStyle = slice_widget.sliceView().sliceViewInteractorStyle()
        interactorStyle.SetActionEnabled(interactorStyle.BrowseSlice, False)
        slice_widget.sliceView().setBackgroundColor(qt.QColor.fromRgbF(1, 1, 1))
        self.slice_frozen = True
        slice_widget.sliceController().setDisabled(self.slice_frozen)

    def unfreezeSlice(self):
        slice_widget = self.lm.sliceWidget(self.logic.slice_direction)
        if not slice_widget:
            return
        interactorStyle = slice_widget.sliceView().sliceViewInteractorStyle()
        interactorStyle.SetActionEnabled(interactorStyle.BrowseSlice, True)
        slice_widget.sliceView().setBackgroundColor(qt.QColor.fromRgbF(0, 0, 0))
        self.slice_frozen = False
        slice_widget.sliceController().setDisabled(self.slice_frozen)

    def clearPoints(self):
        self.actual_remove_click = False
        inc = self._parameterNode.GetNodeReference("fastsamIncludePoints")
        exc = self._parameterNode.GetNodeReference("fastsamExcludePoints")
        if inc: inc.RemoveAllControlPoints()
        self.logic.include_coords = {}
        if exc: exc.RemoveAllControlPoints()
        self.logic.exclude_coords = {}
        self.actual_remove_click = True

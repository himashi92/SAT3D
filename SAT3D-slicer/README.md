# SAT3D for 3D Slicer

Interactive 3D tumour segmentation in [3D Slicer](https://www.slicer.org/) with the
**SAT3D-plus** model: a Swin-based 3D SAM with a critic network that guides refinement.
You prompt with points, brush scribbles, a box and optional free text, then refine the
result with more prompts.

Tested with **3D Slicer 5.6.2** on Windows (CPU). A CUDA GPU is used automatically when
available.

---

## 1. Requirements

| What | Notes |
|---|---|
| 3D Slicer ≥ 5.6 | <https://download.slicer.org> |
| **PyTorch** extension | Install from the Extensions Manager (see step 3) |
| Python packages | `einops`, `timm`, `torchio`, `transformers>=4.40,<4.46`. The module offers to install them on first run. |
| Model weights | `sam_model_dice_best.pth` (~590 MB) and `critic_dice_best.pth` (~65 KB): the SAT3D-plus checkpoints |

> **Why `transformers<4.46`?** Slicer 5.6 ships with PyTorch 2.1, and newer `transformers`
> releases need PyTorch ≥ 2.2. They fail on import with
> `module 'torch.utils._pytree' has no attribute 'register_pytree_node'`.

---

## 2. Add the module to Slicer

1. Get this repository onto your machine (clone or unzip), e.g.
   `C:\Users\<you>\Documents\SAT3D-slicer`.
2. Start 3D Slicer and open **Edit → Application Settings → Modules**.
3. Under **Additional module paths**, click **Add** and select the **`sat3D`** folder inside
   the repository, the one that contains `sat3D.py`:
   ```
   C:\Users\<you>\Documents\SAT3D-slicer\sat3D
   ```
4. Click **OK** and restart Slicer when asked.
5. Open the module from the module menu: **Segmentation → SAT3D**, or search for "SAT3D".

> Shortcut: dragging the `sat3D` folder onto the Slicer window also offers to add it as a
> module path.

---

## 3. Install the dependencies

### PyTorch
1. Open **View → Extensions Manager → Install Extensions**.
2. Search for **PyTorch**, install it and restart Slicer.

On the first **Run / Refine**, SAT3D installs PyTorch itself if it is missing. It asks for
confirmation first.

### Python packages
These are also installed on the first run after you confirm. To install them up front,
paste this into Slicer's Python console (**View → Python Console**):

```python
slicer.util.pip_install('einops timm torchio "transformers>=4.40,<4.46"')
```

---

## 4. Model weights

By default SAT3D looks for the weights in your **Downloads** folder:

```
<Downloads>\sam_model_dice_best.pth
<Downloads>\critic_dice_best.pth
```

To keep them somewhere else (e.g. `SAT3D_Studio\sat3D_plus_model_weights\`), set the paths
in the module under **Advanced → SAT3D weights / Critic weights**. The setting is remembered.

The free-text encoder (PubMedBERT) needs no separate download. Its weights are inside the
SAT3D checkpoint, and the small config/tokenizer files ship with the module in
`sat3D/Resources/TextEncoder/`.

---

## 5. Using SAT3D

1. **Load a scan** (NIfTI, NRRD, DICOM, ...). It is selected as the input volume
   automatically.
2. **Add prompts** for the current segment:
   - **Include points** (key `1`): click inside the lesion.
   - **Exclude points** (key `2`): click on wrongly segmented areas.
   - **Brush scribbles**: click **Brush +** or **Brush −** and paint in any view. Strokes
     are sampled into points. **Erase** removes paint.
   - **Box**: click **Draw Box**, then click two corners in a slice view. Adjust its depth
     with the handles in the other views. A box drawn flat in one view gets the depth set
     under *Advanced → Flat box depth*.
   - **Text** (optional), e.g. `enhancing tumour`. Text must be combined with at least
     one point, scribble or box.
3. Click **Run / Refine**. The first run loads the model and encodes a 128³ region around
   your prompts. Further prompts inside that region refine the result in a few seconds.
   Prompts far away start a new region.
4. **Review**:
   - **Confidence threshold**: re-binarises the last result instantly, without re-running
     the model.
   - **Show model uncertainty (critic)**: blue = relatively certain, red = relatively
     likely mis-segmented. This ranks voxels within the case; it is not an absolute
     probability.
   - **Δ Added / Δ Removed** segments show what changed in the last run.
   - **Undo Last Run** (key `z`), **Clear Segment**, **Clear Prompts** (key `a`).
5. **More lesions**: **Accept & New Segment** (key `n`) starts a new segment with its own
   prompts. Selecting an earlier segment restores its prompts.
6. **Approve** marks a segment as reviewed. The measurements table shows the volume (cm³),
   longest diameter (mm) and approval of every segment.
7. **Save Segmentation...** writes the results (see below). **End Task** restarts Slicer
   for the next case.

### Keyboard shortcuts (active while the module is open)

| Key | Action |
|---|---|
| `1` / `2` | Place include / exclude points |
| `a` | Clear prompts |
| `z` | Undo last run |
| `n` | New segment |

---

## 6. Outputs

Everything for a case goes into a session folder next to the scan:

```
<scan folder>/<scan name>_sat3d_session/
├── session_log.txt                        # OPENED / MODEL-LOADED / RUN / UNDO / APPROVED / SAVED, timestamped
├── <case>_<segment>_<run>_<time>.nii.gz   # each run's mask (if "Auto-save each run" is on)
└── <case>_<segment>_<run>_<time>_prompts.json
```

If the scan has no file on disk (e.g. it was loaded from DICOM and never saved), the
session folder is created in Downloads.

**Save Segmentation...** writes, to a folder you choose:

- `<case>_seg.nii.gz`: one multi-label NIfTI (segment 1 = label 1, segment 2 = label 2, ...)
  with the input scan's geometry.
- `<case>_seg_labels.json`: label → segment name and approval.
- `<case>_<segment>_prompts.json`: the exact points (voxel `d, h, w`), box, text, crop
  region and threshold used for each segment.

---

## 7. Advanced settings

| Setting | Default | Meaning |
|---|---|---|
| SAT3D / Critic weights | Downloads folder | Checkpoint paths |
| Text encoder | `sat3D/Resources/TextEncoder` | PubMedBERT config/tokenizer folder |
| Device | `auto` | `auto` uses CUDA if available, otherwise CPU |
| Seed | 2025 | Random seed set when the model loads |
| ROI margin | 8 voxels | Margin around the prompts when choosing the 128³ crop |
| Scribble stride / max points | 3 voxels / 200 | How densely brush strokes are sampled |
| Flat box depth | 20 slices | Depth given to a box drawn in a single view |
| Auto-save each run | on | Write every run's mask and prompts to the session folder |

Changing the weights, text encoder, device or seed unloads the model. It reloads on the
next run.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| *SAT3D* missing from the module list | Check that the module path points at the `sat3D` folder (the one containing `sat3D.py`) and restart Slicer. Errors appear in **View → Python Console**. |
| "Model weights not found" | Put the `.pth` files in Downloads or set the paths under **Advanced**. |
| "...is missing N weights. Is it the SAT3D-plus checkpoint?" | The file isn't the SAT3D-plus checkpoint (or it is corrupted). |
| `register_pytree_node` error | Your `transformers` is too new for Slicer's PyTorch: `slicer.util.pip_install('"transformers>=4.40,<4.46"')` |
| Runs are slow | Without a GPU, the first run on a case takes ~20 s and refinements a few seconds. Check the device under **Advanced → Model**. |
| "Text alone can't localise a lesion" | Add at least one point, scribble or box. |

---

## 9. Building as a Slicer extension (optional)

The repository is a standard Slicer extension (`CMakeLists.txt` at the root, module in
`sat3D/`). To package it, build against a Slicer build tree:

```bash
cmake -S . -B build -DSlicer_DIR=<path-to-Slicer-build>/Slicer-build
cmake --build build --config Release
```

Model weights are **not** part of the package. Distribute them separately.

---

## Repository layout

```
SAT3D-slicer/
├── CMakeLists.txt                          # extension metadata
├── sample/                                 # example BraTS FLAIR scan
└── sat3D/                                  # <- add this folder as a Slicer module path
    ├── sat3D.py                            # module + GUI
    ├── sat3DLib/
    │   ├── sat3DLogic.py                   # model loading, background runs, display, saving, session log
    │   └── engine.py                       # SAT3D-plus inference engine (ported from SAT3D Studio, no Slicer dependency)
    ├── segment_anything_with_swin_conf_plus/   # SAT3D-plus model code
    ├── networks/                           # critic network
    └── Resources/
        ├── UI/sat3D.ui
        ├── Icons/sat3D.png
        └── TextEncoder/                    # PubMedBERT config + tokenizer (MIT licence)
```

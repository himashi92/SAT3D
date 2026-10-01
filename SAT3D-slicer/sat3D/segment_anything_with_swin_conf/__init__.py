# Only the 3D Swin builder is needed by the Slicer module. The 2D SAM predictor /
# automatic mask generator pull in torchvision, cv2 and the separate
# `segment_anything` package, so they are no longer imported eagerly; import
# them explicitly from their submodules if required.
from .build_samswin3D import build_sam3D, sam_model_registry3D

# Copyright (c) Facebook, Inc. and its affiliates.

# meta architectures
from .modeling.meta_arch import custom_rcnn
from .modeling.meta_arch import phrase_custom_rcnn

# register modules
from .modeling.roi_heads import roi_emb_heads, res5_roi_heads

# from .modeling.roi_heads import text_roi_heads

# datasets
from .data.datasets import coco_ovd
from .data.datasets import ovad

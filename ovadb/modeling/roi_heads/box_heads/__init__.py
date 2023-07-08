from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from .detic_fast_rcnn import DeticFastRCNNOutputLayers
from .phrase_fast_rcnn import PhraseFastRCNNOutputLayers
from .emb_fast_rcnn import EmbeddingFastRCNNOutputLayers
from .vlplm_fast_rcnn import VLPLMFastRCNNOutputLayers


def build_box_predictor(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    BOX_EMBEDDING_PREDICTORS = {
        "FastRCNNOutputLayers": FastRCNNOutputLayers,
        "DeticFastRCNNOutputLayers": DeticFastRCNNOutputLayers,
        "PhraseFastRCNNOutputLayers": PhraseFastRCNNOutputLayers,
        "EmbeddingFastRCNNOutputLayers": EmbeddingFastRCNNOutputLayers,
        "VLPLMFastRCNNOutputLayers": VLPLMFastRCNNOutputLayers,
    }
    return BOX_EMBEDDING_PREDICTORS[name](cfg, input_shape)

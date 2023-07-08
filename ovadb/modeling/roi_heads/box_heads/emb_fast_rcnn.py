import logging
from typing import Dict, List, Tuple, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import (
    fast_rcnn_inference,
    fast_rcnn_inference_single_image,
    FastRCNNOutputLayers,
    _log_classification_stats,
)
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

from .zero_shot_classifier import ZeroShotClassifier
from ovadb.modeling.logged_module import LoggedModule

__all__ = ["EmbeddingFastRCNNOutputLayers"]

logger = logging.getLogger(__name__)


class EmbeddingFastRCNNOutputLayers(FastRCNNOutputLayers, LoggedModule):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        cls_score=None,
        use_zeroshot_cls: bool = False,
        two_layer_bbox_predictor: bool = False,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            **kwargs,
        )

        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        num_inputs = (
            input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        )
        # prediction layer for num_classes foreground classes and one background class (hence + 1)

        # Add the embedding based layer
        self.use_zeroshot_cls = use_zeroshot_cls
        if self.use_zeroshot_cls:
            del self.cls_score
            assert cls_score is not None
            self.cls_score = cls_score

        # Make a deeper bbox predictor to separate the semantics from the localizatio
        # idea taken from Detic
        if two_layer_bbox_predictor:
            del self.bbox_pred
            self.bbox_pred = nn.Sequential(
                nn.Linear(num_inputs, num_inputs),
                nn.ReLU(inplace=True),
                nn.Linear(num_inputs, 4),
            )
            weight_init.c2_xavier_fill(self.bbox_pred[0])
            nn.init.normal_(self.bbox_pred[-1].weight, std=0.001)
            nn.init.constant_(self.bbox_pred[-1].bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update(
            {
                # fmt: on
                "use_zeroshot_cls": cfg.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS,
                "two_layer_bbox_predictor": cfg.MODEL.ROI_BOX_HEAD.TWO_LAYER_BBOX_PRED,
            }
        )
        if ret["use_zeroshot_cls"]:
            ret["cls_score"] = ZeroShotClassifier(cfg, input_shape)
        return ret

    def forward(self, x, classifier_info=None):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        proposal_deltas = self.bbox_pred(x)
        scores = self.cls_score(x)

        return scores, proposal_deltas

    def inference(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        predictions = (predictions[0], predictions[1])
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def inference_on_box(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = tuple([x.pred_boxes.tensor for x in proposals])
        predictions = (predictions[0], predictions[1])
        # boxes_pred = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        test_score_thresh = 0.001
        test_nms_thresh = 0.8
        test_topk_per_image = int(max([len(b) for b in boxes]) * 1.1)
        instances, indx = fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            test_score_thresh,
            test_nms_thresh,
            self.test_topk_per_image,
        )
        for instance, prop, idx in zip(instances, proposals, indx):
            if prop.has("gt_classes"):
                gt_classes = prop.get("gt_classes")
                gt_classes = gt_classes[idx]
                instance.gt_classes = gt_classes

        return instances, indx

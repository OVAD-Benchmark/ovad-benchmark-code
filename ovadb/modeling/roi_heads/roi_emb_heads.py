import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, ROIHeads, Res5ROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)

from .box_heads import build_box_predictor
from .attribute_heads import build_attribute_predictor


class SampleAllROIHeads(ROIHeads):
    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Based on label_and_sample_proposals from roi_heads.py in detectron
        with the modification of copying all fields not only gt_ label

        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # matched_idxs (Tensor[int64]): a vector of length N, where matches[i] is a matched
            #       ground-truth index in [0, M)
            # matched_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
            #       whether a prediction is a true or false positive or ignored
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            # sampled_targets = targets_per_image[target_index]
            # sampled_targets.gt_classes = gt_classes

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    # Here is the change with default it copies all fields not only gt_ label
                    if not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])

            fg_classes = torch.ones_like(gt_classes)
            fg_classes[gt_classes == self.num_classes] = 0
            proposals_per_image.set("fg_proposal", fg_classes)

            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt


@ROI_HEADS_REGISTRY.register()
class EmbeddingRes5ROIHeads(Res5ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    See :paper:`ResNet` Appendix A.
    """

    @configurable
    def __init__(
        self,
        *,
        attribute_head: Optional[nn.Module] = None,
        output_shape: Optional[int] = 0,
        **kwargs,
    ):
        """
        Added attribute head
        """
        super().__init__(**kwargs)
        self.output_shape = output_shape

        self.attribute_on = attribute_head is not None
        if self.attribute_on:
            self.attribute_head = attribute_head
            self.attribute_on = False  # deactivate attribute head

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        attribute_on = cfg.MODEL.ATTRIBUTE_ON

        ret["res5"], out_channels = cls._build_res5_block(cfg)
        ret["box_predictor"] = build_box_predictor(cfg, input_shape=out_channels)
        ret["output_shape"] = out_channels

        if attribute_on:
            ret["attribute_head"] = build_attribute_predictor(
                cfg,
                input_shape=out_channels,
            )

        return ret

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Based on label_and_sample_proposals from roi_heads.py in detectron
        with the modification of copying all fields not only gt_ label

        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # matched_idxs (Tensor[int64]): a vector of length N, where matches[i] is a matched
            #       ground-truth index in [0, M)
            # matched_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
            #       whether a prediction is a true or false positive or ignored
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            # sampled_targets = targets_per_image[target_index]
            # sampled_targets.gt_classes = gt_classes

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    # Here is the change with default it copies all fields not only gt_ label
                    if not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])

            fg_classes = torch.ones_like(gt_classes)
            fg_classes[gt_classes == self.num_classes] = 0
            proposals_per_image.set("fg_proposal", fg_classes)

            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes")
        if not instances[0].has("pred_classes"):
            if isinstance(features, Dict):
                features = [features[f] for f in self.in_features]
            instance_boxes = [x.pred_boxes for x in instances]
            box_features = self._shared_roi_transform(features, instance_boxes)
            predictions = self.box_predictor(box_features.mean(dim=[2, 3]))
            new_instances, _ = self.box_predictor.inference_on_box(
                predictions, instances
            )
            if isinstance(new_instances[0], list):
                instances = [inst[0] for inst in new_instances]
            else:
                instances = new_instances

        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            if isinstance(features, Dict):
                features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(
                features, [inst.pred_boxes for inst in instances]
            )
            instances = self.mask_head(x, instances)
        if self.attribute_on:
            if isinstance(features, Dict):
                features = [features[f] for f in self.in_features]

            x = self._shared_roi_transform(
                features, [inst.pred_boxes for inst in instances]
            )
            x = self.box_predictor.cls_score.linear(x.mean(dim=[2, 3]))
            instances = self.attribute_head(x, instances)

        return instances

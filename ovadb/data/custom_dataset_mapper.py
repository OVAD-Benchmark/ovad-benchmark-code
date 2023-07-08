import os
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch
import pycocotools.mask as mask_util
import dill as pickle
import random

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data.detection_utils import transform_keypoint_annotations
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.structures import Keypoints, PolygonMasks, BitMasks
from fvcore.transforms.transform import TransformList

from ovadb.data.custom_build_augmentation import build_custom_augmentation
from ovadb.data.data_utils import annotations_to_instances

__all__ = ["CustomDatasetMapper"]


class CustomDatasetMapper(DatasetMapper):
    @configurable
    def __init__(
        self,
        is_train: bool,
        with_ann_type=False,
        dataset_ann=[],
        use_diff_bs_size=False,
        dataset_augs=[],
        is_debug=False,
        caption_features="",
        phrase_features="",
        neg_captions=0,
        neg_phrases=0,
        **kwargs
    ):
        """
        add image labels
        """
        self.with_ann_type = with_ann_type
        self.dataset_ann = dataset_ann
        self.use_diff_bs_size = use_diff_bs_size
        if self.use_diff_bs_size and is_train:
            self.dataset_augs = [T.AugmentationList(x) for x in dataset_augs]
        self.is_debug = is_debug

        # Load in memory either caption or phrase noun features
        if os.path.isfile(caption_features):
            self.caption_features = pickle.load(open(caption_features, "rb"))
            self.neg_captions = neg_captions
        else:
            self.caption_features = None
            self.neg_captions = 0

        if os.path.isfile(phrase_features):
            self.phrase_features = pickle.load(open(phrase_features, "rb"))
            self.neg_phrases = neg_phrases
            self.caption_features = self.phrase_features['captions']
            self.neg_captions = neg_captions
        else:
            self.phrase_features = None
            self.neg_phrases = 0
        super().__init__(is_train, **kwargs)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        ret.update(
            {
                "with_ann_type": cfg.WITH_ANNOTATION_TYPE,
                "dataset_ann": cfg.DATALOADER.DATASET_ANN,
                "use_diff_bs_size": cfg.DATALOADER.USE_DIFF_BS_SIZE,
                "is_debug": cfg.IS_DEBUG,
                "caption_features": cfg.DATALOADER.CAPTION_FEATURES,
                "neg_captions": cfg.DATALOADER.NUM_NEG_CAPTIONS,
                "phrase_features": cfg.DATALOADER.PHRASE_FEATURES,
                "neg_phrases": cfg.DATALOADER.NUM_NEG_PHRASES,
            }
        )
        if ret["use_diff_bs_size"] and is_train:
            if cfg.INPUT.CUSTOM_AUG == "EfficientDetResizeCrop":
                dataset_scales = cfg.DATALOADER.DATASET_INPUT_SCALE
                dataset_sizes = cfg.DATALOADER.DATASET_INPUT_SIZE
                ret["dataset_augs"] = [
                    build_custom_augmentation(cfg, True, scale, size)
                    for scale, size in zip(dataset_scales, dataset_sizes)
                ]
            else:
                assert cfg.INPUT.CUSTOM_AUG == "ResizeShortestEdge"
                min_sizes = cfg.DATALOADER.DATASET_MIN_SIZES
                max_sizes = cfg.DATALOADER.DATASET_MAX_SIZES
                ret["dataset_augs"] = [
                    build_custom_augmentation(cfg, True, min_size=mi, max_size=ma)
                    for mi, ma in zip(min_sizes, max_sizes)
                ]
        else:
            ret["dataset_augs"] = []

        return ret

    def __call__(self, dataset_dict):
        """
        include image labels
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        if "file_name" in dataset_dict:
            ori_image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        else:
            ori_image, _, _ = self.tar_dataset[dataset_dict["tar_index"]]
            ori_image = utils._apply_exif_orientation(ori_image)
            ori_image = utils.convert_PIL_to_numpy(ori_image, self.image_format)
        utils.check_image_size(dataset_dict, ori_image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        if self.is_debug:
            dataset_dict["dataset_source"] = 0

        not_full_labeled = (
            "dataset_source" in dataset_dict
            and self.with_ann_type
            and self.dataset_ann[dataset_dict["dataset_source"]] != "box"
        )

        aug_input = T.AugInput(copy.deepcopy(ori_image), sem_seg=sem_seg_gt)
        if self.use_diff_bs_size and self.is_train:
            transforms = self.dataset_augs[dataset_dict["dataset_source"]](aug_input)
        else:
            transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            all_annos = [
                (
                    utils.transform_instance_annotations(
                        obj,
                        transforms,
                        image_shape,
                        keypoint_hflip_indices=self.keypoint_hflip_indices,
                    ),
                    obj.get("iscrowd", 0),
                )
                for obj in dataset_dict.pop("annotations")
            ]
            annos = [ann[0] for ann in all_annos if ann[1] == 0]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            del all_annos
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if self.with_ann_type:
            dataset_dict["pos_category_ids"] = dataset_dict.get("pos_category_ids", [])
            dataset_dict["ann_type"] = self.dataset_ann[dataset_dict["dataset_source"]]
        if self.is_debug and (
            ("pos_category_ids" not in dataset_dict)
            or (dataset_dict["pos_category_ids"] == [])
        ):
            dataset_dict["pos_category_ids"] = [
                x for x in sorted(set(dataset_dict["instances"].gt_classes.tolist()))
            ]

        if self.caption_features:
            caption_rep = self.caption_features.get(dataset_dict["image_id"], None)
            captions = []
            cap_features = []
            if caption_rep is not None:
                for (cap, rep) in caption_rep:
                    captions.append(cap)
                    cap_features.append(rep)
                cap_features = np.stack(cap_features, axis=0)
            dataset_dict["captions"] = captions
            dataset_dict["cap_features"] = cap_features
            if self.neg_captions > 0:
                set_ids = set(self.caption_features.keys()).difference(
                    {dataset_dict["image_id"]}
                )
                set_ids = random.sample(set_ids, self.neg_captions)
                neg_captions = []
                neg_cap_features = []
                for n_id in set_ids:
                    caption = random.choice(self.caption_features.get(n_id, None))
                    neg_captions.append(caption[0])
                    neg_cap_features.append(caption[1])
                neg_cap_features = np.stack(neg_cap_features, axis=0)
                dataset_dict["neg_cap_img_ids"] = set_ids
                dataset_dict["neg_captions"] = neg_captions
                dataset_dict["neg_cap_features"] = neg_cap_features

        if self.phrase_features and "poc" in dataset_dict.keys():
            # select the positive part-of-captions
            pos_nph, pos_nouns, pos_ncomp = [], [], []
            for cap_dict in dataset_dict["poc"]:
                for poc in cap_dict["poc"]:
                    pos_nph.append(poc["nph"])
                    pos_nouns += poc["nouns"]
                    pos_ncomp += poc["ncomp"]
            pos_nph, pos_nouns, pos_ncomp = set(pos_nph), set(pos_nouns), set(pos_ncomp) 
            dataset_dict["pos_nph"] = list(pos_nph)
            dataset_dict["pos_nouns"] = list(pos_nouns)
            dataset_dict["pos_ncomp"] = list(pos_ncomp)
            
            text_rep_dict = {}
            for phn in pos_nph:
                text_rep_dict[phn] = self.phrase_features['noun_phrases'].get(phn, None)
            for text in pos_nouns.union(pos_ncomp):
                text_rep_dict[text] = self.phrase_features['text_words'].get(text, None)

            # select negative part-of-captions
            neg_phrases = set(self.phrase_features['noun_phrases'].keys()).difference(set(pos_nph))
            neg_phrases = random.sample(neg_phrases, self.neg_phrases)
            for phn in neg_phrases:
                text_rep_dict[phn] = self.phrase_features['noun_phrases'].get(phn, None)

            neg_nouns = set(self.phrase_features['nouns']).difference(set(pos_nouns))
            neg_nouns = random.sample(neg_nouns, self.neg_phrases)
            neg_ncomp = set(self.phrase_features['ncomp']).difference(set(pos_ncomp))
            neg_ncomp = random.sample(neg_ncomp, self.neg_phrases)
            for text in set(neg_nouns).union(set(neg_ncomp)):
                text_rep_dict[text] = self.phrase_features['text_words'].get(text, None)

            dataset_dict["neg_nph"] = list(neg_phrases)
            dataset_dict["neg_nouns"] = list(neg_nouns)
            dataset_dict["neg_ncomp"] = list(neg_ncomp)

            dataset_dict["text_rep_dict"] = text_rep_dict

        return dataset_dict

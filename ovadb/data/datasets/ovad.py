"""
Copyright (c) 2022
All rights reserved.
For full license text see https://ovad-benchmark.github.io/
By Maria A. Bravo

This file contains functions to parse captions and builds the poc training json
"""

"""
This file contains functions to parse OVAD-format annotations into dicts in "Detectron2 format".
"""
import io
import os
import json
import shutil
import logging
import datetime
import contextlib
import numpy as np
from fvcore.common.file_io import PathManager, file_lock

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta

from ovadb.data.datasets.register_data import register_custom_instances
from datasets.ovad.ovad import OVAD

logger = logging.getLogger(__name__)


def convert_to_coco_extended_json(dataset_name, output_file, allow_cached=True):
    """
    Adds extra fields in the json file. To include attributes
    """

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data

    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.warning(
                f"Using previously cached COCO format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            logger.info(
                f"Converting annotations of dataset '{dataset_name}' to COCO format ...)"
            )
            coco_dict = convert_to_coco_extended_dict(dataset_name)

            logger.info(f"Caching COCO format annotations at '{output_file}' ...")
            tmp_file = output_file + ".tmp"
            with PathManager.open(tmp_file, "w") as f:
                json.dump(coco_dict, f)
            shutil.move(tmp_file, output_file)


def convert_to_coco_extended_dict(dataset_name):
    """
    Adds other fields in the json file
    """

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {
            v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()
        }
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[
            contiguous_id
        ]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    # nouns
    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    # attributes
    attributes = [
        {
            "id": idx,
            "name": att,
            "type": att.split(":")[0],
            "parent_type": [
                key
                for key, val in metadata.att_parent_type.items()
                if att.split(":")[0] in val
            ][0],
            "freq_set": [
                key for key, val in metadata.attribute_head_tail.items() if att in val
            ][0],
            "is_has_att": [
                key
                for key, val in metadata.attribute_hierarchy["is_has_att"].items()
                if att.split(":")[0] in val
            ][0],
        }
        for idx, att in enumerate(metadata.attribute_classes)
    ]

    logger.info("Converting dataset dicts into COCO extended format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": int(image_dict["width"]),
            "height": int(image_dict["height"]),
            "file_name": str(image_dict["file_name"]),
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict.get("annotations", [])
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format for axis-align and XYWHA for rotated
            bbox = annotation["bbox"]
            if isinstance(bbox, np.ndarray):
                if bbox.ndim != 1:
                    raise ValueError(
                        f"bbox has to be 1-dimensional. Got shape={bbox.shape}."
                    )
                bbox = bbox.tolist()
            if len(bbox) not in [4, 5]:
                raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
            from_bbox_mode = annotation["bbox_mode"]
            to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
            bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)

            # COCO requirement: instance area
            if "segmentation" in annotation:
                # Computing areas for instances by counting the pixels
                segmentation = annotation["segmentation"]
                # TODO: check segmentation type: RLE, BinaryMask or Polygon
                if isinstance(segmentation, list):
                    polygons = PolygonMasks([segmentation])
                    area = polygons.area()[0].item()
                elif isinstance(segmentation, dict):  # RLE
                    area = mask_util.area(segmentation).item()
                else:
                    raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
            else:
                # Computing areas using bounding boxes
                if to_bbox_mode == BoxMode.XYWH_ABS:
                    bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
                    area = Boxes([bbox_xy]).area()[0].item()
                else:
                    area = RotatedBoxes([bbox]).area()[0].item()

            if "keypoints" in annotation:
                keypoints = annotation["keypoints"]  # list[int]
                for idx, v in enumerate(keypoints):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # For COCO format consistency we substract 0.5
                        # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
                        keypoints[idx] = v - 0.5
                if "num_keypoints" in annotation:
                    num_keypoints = annotation["num_keypoints"]
                else:
                    num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

            if "att_vec" in annotation:
                att_vec = annotation["att_vec"]  # array
                if len(att_vec) == len(attributes):
                    att_vec = att_vec.astype(int).tolist()
                else:
                    att_vec = np.zeros(len(attributes), dtype=int) - 1
                    assert (
                        "pos_att" in image_dict.keys()
                        and "neg_att" in image_dict.keys()
                    )
                    pos_att = image_dict["pos_att"][annotation["id"]]
                    neg_att = image_dict["neg_att"][annotation["id"]]
                    for att in pos_att:
                        att_idx = metadata.att2idx[att]
                        att_vec[att_idx] = 1
                    for att in neg_att:
                        att_idx = metadata.att2idx[att]
                        att_vec[att_idx] = 0
                    att_vec = att_vec.astype(int).tolist()

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = float(area)
            coco_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
            coco_annotation["category_id"] = int(
                reverse_id_mapper(annotation["category_id"])
            )

            # Add optional fields
            if "keypoints" in annotation:
                coco_annotation["keypoints"] = keypoints
                coco_annotation["num_keypoints"] = num_keypoints

            if "segmentation" in annotation:
                seg = coco_annotation["segmentation"] = annotation["segmentation"]
                if isinstance(seg, dict):  # RLE
                    counts = seg["counts"]
                    if not isinstance(counts, str):
                        # make it json-serializable
                        seg["counts"] = counts.decode("ascii")

            if "att_vec" in annotation:
                coco_annotation["att_vec"] = att_vec

            coco_annotations.append(coco_annotation)

    logger.info(
        "Conversion finished, "
        f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Generated COCO Attributes json file for Detectron2.",
    }
    coco_dict = {
        "info": info,
        "images": coco_images,
        "categories": categories,
        "attributes": attributes,
        "licenses": None,
    }
    if len(coco_annotations) > 0:
        coco_dict["annotations"] = coco_annotations
    return coco_dict


def _get_ovad_meta(dataset_name, ann_file):
    metadata = _get_coco_instances_meta()
    ann_file = PathManager.get_local_path(ann_file)
    with contextlib.redirect_stdout(io.StringIO()):
        ovad_api = OVAD(ann_file)

    # Make the dictionaries for evaluator of attributes
    att2idx = {}
    idx2att = {}
    attr_type = {}
    attr_parent_type = {}
    attribute_head_tail = {"head": set(), "medium": set(), "tail": set()}

    for att in ovad_api.atts.values():
        att2idx[att["name"]] = att["id"]
        idx2att[att["id"]] = att["name"]

        if att["type"] not in attr_type.keys():
            attr_type[att["type"]] = set()
        attr_type[att["type"]].add(att["name"])

        if att["parent_type"] not in attr_parent_type.keys():
            attr_parent_type[att["parent_type"]] = set()
        attr_parent_type[att["parent_type"]].add(att["type"])

        attribute_head_tail[att["freq_set"]].add(att["name"])

    attr_type = {key: list(val) for key, val in attr_type.items()}
    attr_parent_type = {key: list(val) for key, val in attr_parent_type.items()}
    attribute_head_tail = {key: list(val) for key, val in attribute_head_tail.items()}

    attribute_list = list(att2idx.keys())
    attCount = {att: 0 for att in attribute_list}

    metadata["attribute_classes"] = attribute_list
    metadata["att2idx"] = att2idx
    metadata["idx2att"] = idx2att
    metadata["att_base_novel"] = {}
    metadata["att_type"] = attr_type
    metadata["att_parent_type"] = attr_parent_type
    evaluator_type = "attribute"
    if "boxann" in dataset_name:
        evaluator_type += "_boxann"
    metadata["evaluator_type"] = evaluator_type
    metadata["attribute_head_tail"] = attribute_head_tail

    # Add object information in metadata
    cat_ids = list(ovad_api.cats.keys())
    cats = list(ovad_api.cats.values())
    # The categories in a custom json file may not be sorted.
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    metadata["thing_classes"] = thing_classes
    # In COCO, certain category ids are artificially removed,
    # and by convention they are always ignored.
    # We deal with COCO's id issue and translate
    # the category ids to contiguous ids in [0, 80).

    # It works by looking at the "categories" field in the json, therefore
    # if users' own json also have incontiguous ids, we'll
    # apply this mapping as well but print a warning.
    id_map = {v: i for i, v in enumerate(cat_ids)}
    metadata["thing_dataset_id_to_contiguous_id"] = id_map

    return metadata


_PREDEFINED_SPLITS_OVAD = {
    "ovad2000": "ovad/ovad2000.json",
    "ovad2000_boxann": "ovad/ovad2000.json",
}

for dataset_name, json_file in _PREDEFINED_SPLITS_OVAD.items():
    image_root = "coco/val2017"
    extra_annotation_keys = ["att_vec", "id"]
    register_custom_instances(
        dataset_name,
        _get_ovad_meta(dataset_name, os.path.join("datasets", json_file)),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
        extra_annotation_keys,
    )

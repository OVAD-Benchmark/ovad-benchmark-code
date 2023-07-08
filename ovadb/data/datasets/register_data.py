"""
Copyright (c) 2022
All rights reserved.
For full license text see https://ovad-benchmark.github.io/
By Maria A. Bravo

This file contains functions to parse LVIS-format annotations into dicts with custom keys in the
"Detectron2 format".
This script was based on:
https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/lvis.py
"""

import logging
import os
import contextlib
import io

import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger(__name__)

__all__ = ["load_custom_json", "register_custom_instances"]


def register_custom_instances(
    name,
    metadata,
    json_file,
    image_root,
    extra_annotation_keys=[],
    evaluator_type="lvis",
):
    """
    Register dataset not registeres before
    """
    if name not in DatasetCatalog and name not in MetadataCatalog:
        DatasetCatalog.register(
            name,
            lambda: load_custom_json(
                json_file, image_root, metadata, name, extra_annotation_keys
            ),
        )
        evaluator_type_meta = metadata.pop("evaluator_type", None)
        evaluator_type = evaluator_type_meta if evaluator_type_meta else evaluator_type
        MetadataCatalog.get(name).set(
            json_file=json_file,
            image_root=image_root,
            evaluator_type=evaluator_type,
            **metadata,
        )


def load_custom_json(
    json_file, image_root, metadata, dataset_name=None, extra_annotation_keys=None
):
    """
    Load a json file in LVIS's / COCO's annotation format.
    Args:
        json_file (str): full path to the LVIS json annotation file.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., "lvis_v0.5_train").
            If provided, this function will put "thing_classes" into the metadata
            associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "bbox", "bbox_mode", "category_id",
            "segmentation"). The values for these keys will be returned as-is.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from lvis import LVIS

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    # Make a continuous indexing of ids
    catid2contid = metadata.get(
        "thing_dataset_id_to_contiguous_id",
        {
            x["id"]: i
            for i, x in enumerate(
                sorted(lvis_api.dataset["categories"], key=lambda x: x["id"])
            )
        },
    )
    if len(lvis_api.dataset["categories"]) == 1203:
        for x in lvis_api.dataset["categories"]:
            assert catid2contid[x["id"]] == x["id"] - 1
    img_ids = sorted(lvis_api.imgs.keys())
    imgs = lvis_api.load_imgs(img_ids)
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    # Sanity check that each annotation has a unique id
    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(
        ann_ids
    ), "Annotation ids in '{}' are not unique".format(json_file)

    imgs_anns = list(zip(imgs, anns))
    logger.info(
        "Loaded {} images in the LVIS format from {}".format(len(imgs_anns), json_file)
    )

    # Here extra annotation keys should be specify
    if extra_annotation_keys:
        logger.info(
            "The following extra annotation keys will be loaded: {} ".format(
                extra_annotation_keys
            )
        )
    else:
        extra_annotation_keys = []

    # Make the dicts for each image
    dataset_dicts = []
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        if "file_name" in img_dict:
            file_name = img_dict["file_name"]
            if img_dict["file_name"].startswith("COCO"):
                file_name = file_name[-16:]
            record["file_name"] = os.path.join(image_root, file_name)
        elif "coco_url" in img_dict:
            # e.g., http://images.cocodataset.org/train2017/000000391895.jpg
            split_folder, file_name = img_dict["coco_url"].split("/")[-2:]
            record["file_name"] = os.path.join(image_root, split_folder, file_name)

        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get(
            "not_exhaustive_category_ids", []
        )
        # image label
        record["neg_category_ids"] = [
            catid2contid[x] for x in img_dict.get("neg_category_ids", [])
        ]
        record["pos_category_ids"] = [
            catid2contid[x] for x in img_dict.get("pos_category_ids", [])
        ]
        # caption
        record["captions"] = img_dict.get("captions", [])

        # other keys
        found_extra_keys = {key: False for key in extra_annotation_keys}
        for key in extra_annotation_keys:
            if key in img_dict.keys():
                found_extra_keys[key] = True
                record[key] = img_dict.get(key, [])

        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.
            assert anno["image_id"] == image_id
            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
            # LVIS data loader can be used to load COCO dataset categories. In this case `meta`
            # variable will have a field with COCO-specific category mapping.
            if (
                dataset_name is not None
                and "thing_dataset_id_to_contiguous_id" in metadata
            ):
                obj["category_id"] = metadata["thing_dataset_id_to_contiguous_id"][
                    anno["category_id"]
                ]
            else:
                obj["category_id"] = (
                    anno["category_id"] - 1
                )  # Convert 1-indexed to 0-indexed

            if "segmentation" in anno:
                segm = anno["segmentation"]  # list[list[float]]
                # filter out invalid polygons (< 3 points)
                valid_segm = [
                    poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6
                ]
                assert len(segm) == len(
                    valid_segm
                ), "Annotation contains an invalid polygon with < 3 points"
                assert len(segm) > 0
                obj["segmentation"] = segm

            for extra_ann_key in extra_annotation_keys:
                if extra_ann_key in anno.keys():
                    found_extra_keys[extra_ann_key] = True
                obj[extra_ann_key] = anno.get(extra_ann_key, [])

            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    if len(found_extra_keys) > 0:
        for key, found in found_extra_keys.items():
            print("Key {} found {}".format(key, found))

    return dataset_dicts


def load_coco_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        if not hasattr(meta, "thing_classes"):
            # The categories in a custom json file may not be sorted.
            meta.thing_classes = thing_classes
        else:
            assert len(meta.thing_classes) == len(thing_classes), (
                len(meta.thing_classes),
                len(thing_classes),
            )

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        if not hasattr(meta, "thing_dataset_id_to_contiguous_id"):
            # It works by looking at the "categories" field in the json, therefore
            # if users' own json also have incontiguous ids, we'll
            # apply this mapping as well but print a warning.
            if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
                if "coco" not in dataset_name:
                    logger.warning(
                        """
    Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
    """
                    )
            id_map = {v: i for i, v in enumerate(cat_ids)}
            meta.thing_dataset_id_to_contiguous_id = id_map
        else:
            id_map = meta.thing_dataset_id_to_contiguous_id

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(
            ann_ids
        ), "Annotation ids in '{}' are not unique!".format(json_file)

    imgs_anns = list(zip(imgs, anns))
    logger.info(
        "Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file)
    )

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (
        extra_annotation_keys or []
    )

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert (
                anno.get("ignore", 0) == 0
            ), '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [
                        poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6
                    ]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts

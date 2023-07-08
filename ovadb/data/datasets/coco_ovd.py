import os
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES, _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog
from ovadb.data.datasets.register_data import register_custom_instances, load_coco_json
from ovadb.data.datasets.utils import (
    categories_base,
    categories_novel32,
    categories_novel17,
)


def _get_metadata(cat):
    if cat == "all":
        return _get_builtin_metadata("coco")
    else:
        id_to_name = {}
        if "base" in cat:
            id_to_name.update({x["id"]: x["name"] for x in categories_base})
        if "novel" in cat:
            if "novel32" in cat:
                id_to_name.update({x["id"]: x["name"] for x in categories_novel32})
            else:
                id_to_name.update({x["id"]: x["name"] for x in categories_novel17})

        assert len(id_to_name) > 0

        thing_dataset_id_to_contiguous_id = {
            x: i for i, x in enumerate(sorted(id_to_name))
        }
        thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
        return {
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
            "thing_classes": thing_classes,
        }


_PREDEFINED_SPLITS_COCO = {
    # "coco_train2017": (
    #     "coco/train2017",
    #     "coco/annotations/instances_train2017.json",
    #     "all",
    # ),
    "coco_train2017_ovd_base": (
        "coco/train2017",
        "coco/annotations/instances_train2017_base.json",
        "base",
    ),
    "coco_val2017_ovd17_g": (
        "coco/val2017",
        "coco/annotations/instances_val2017_base_novel17.json",
        "base_novel17",
    ),
    "coco_val2017_ovd32_g": (
        "coco/val2017",
        "coco/annotations/instances_val2017.json",
        "all",
    ),
}


def register_custom_coco_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    if name not in DatasetCatalog and name not in MetadataCatalog:

        # 1. register a function which returns dicts
        DatasetCatalog.register(
            name, lambda: load_coco_json(json_file, image_root, name)
        )

        # 2. Optionally, add metadata about this dataset,
        # since they might be useful in evaluation, visualization or logging
        MetadataCatalog.get(name).set(
            json_file=json_file,
            image_root=image_root,
            evaluator_type="coco",
            **metadata
        )


for key, (image_root, json_file, cat) in _PREDEFINED_SPLITS_COCO.items():
    register_custom_coco_instances(
        key,
        _get_metadata(cat),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )

_CUSTOM_SPLITS_COCO = {
    # captions
    "coco_captions_train2017": (
        "coco/train2017/",
        "coco/annotations/captions_train2017_categories.json",
        [],
    ),
    # poc
    "coco_nouns_train2017": (
        "coco/train2017/",
        "coco/annotations/poc_captions_train2017.json",
        ["poc"],
    ),
    # # nouns + noun phrases
    # "coco_nouns_np_train2017": (
    #     "coco/train2017/",
    #     "coco/annotations/nouns_np_train2017.json",
    # ),
    # # nouns + noun complements
    # "coco_nouns_nc_train2017": (
    #     "coco/train2017/",
    #     "coco/annotations/nouns_nc_train2017.json",
    # ),
    # # captions + nouns
    # "coco_captions_nouns_train2017": (
    #     "coco/train2017/",
    #     "coco/annotations/captions_nouns_train2017.json",
    # ),
    # # captions + nouns + noun phrases
    # "coco_captions_nouns_np_train2017": (
    #     "coco/train2017/",
    #     "coco/annotations/captions_nouns_np_train2017.json",
    # ),
    # # captions + nouns + noun complements
    # "coco_captions_nouns_nc_train2017": (
    #     "coco/train2017/",
    #     "coco/annotations/captions_nouns_nc_train2017.json",
    # ),
}

for key, (image_root, json_file, extra_annotation_keys) in _CUSTOM_SPLITS_COCO.items():
    if "panoptic" in key:
        coco_meta_name = "coco_panoptic_standard"
    else:
        coco_meta_name = "coco"
    register_custom_instances(
        key,
        _get_builtin_metadata(coco_meta_name),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
        extra_annotation_keys,
    )

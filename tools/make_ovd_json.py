"""
Copyright (c) 2022
All rights reserved.
For full license text see https://ovad-benchmark.github.io/
By Maria A. Bravo

This file contains functions to parse json annotation files and builds the training json
"""
import os
import argparse
import json
from collections import defaultdict
import sys

sys.path.insert(0, os.getcwd())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path", default="datasets/coco/annotations/instances_val2017.json"
    )
    parser.add_argument(
        "--cat_path", default="datasets/coco/annotations/instances_val2017.json"
    )
    parser.add_argument("--save_json_path", default="")
    parser.add_argument("--base_novel", default="all")
    parser.add_argument("--convert_caption", action="store_true")
    args = parser.parse_args()

    # Load all categories from file
    print("Loading", args.cat_path)
    cat = json.load(open(args.cat_path, "r"))["categories"]

    # load annotation file
    print("Loading", args.json_path)
    data = json.load(open(args.json_path, "r"))

    # if caption file
    if args.convert_caption:
        num_caps = 0
        caps = defaultdict(list)
        for x in data["annotations"]:
            caps[x["image_id"]].append(x["caption"])
        for x in data["images"]:
            x["captions"] = caps[x["id"]]
            num_caps += len(x["captions"])
        print("# captions", num_caps)
        data["annotations"] = []
        save_json_path = os.path.join(
            os.path.dirname(args.json_path),
            "{set_name}_categories.json".format(
                set_name=os.path.basename(args.json_path).replace(".json", ""),
            ),
        )

    # if instance file
    else:
        if args.base_novel != "all":
            if "/coco/" in args.json_path:
                from ovadb.data.datasets.coco_ovd import (
                    categories_base,
                    categories_novel17,
                    categories_novel32,
                )

                valid_ids = []
                if "base" in args.base_novel:
                    valid_ids.extend([x["id"] for x in categories_base])
                if "novel17" in args.base_novel:
                    valid_ids.extend([x["id"] for x in categories_novel17])
                if "novel32" in args.base_novel:
                    valid_ids.extend([x["id"] for x in categories_novel32])

                # filter annotation file
                filtered_images = []
                filtered_annotations = []
                useful_image_ids = set()

                for ann in data["annotations"]:
                    if ann["category_id"] in valid_ids:
                        filtered_annotations.append(ann)
                        useful_image_ids.add(ann["image_id"])

                for img in data["images"]:
                    if img["id"] in useful_image_ids:
                        filtered_images.append(img)

            data["annotations"] = filtered_annotations
            data["images"] = filtered_images
            data["categories"] = [c for c in cat if c["id"] in valid_ids]

        else:
            data["categories"] = cat

        save_json_path = os.path.join(
            os.path.dirname(args.json_path),
            "{set_name}_{base_novel}.json".format(
                set_name=os.path.basename(args.json_path).replace(".json", ""),
                base_novel=args.base_novel,
            ),
        )

    print("Total images", len(data["images"]))
    print("Total annotations", len(data["annotations"]))
    print("Total categories", len(data["categories"]))

    # save modified json data
    if args.save_json_path != "":
        save_json_path = args.save_json_path
    print("Saving to", save_json_path)

    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    json.dump(data, open(save_json_path, "w"))

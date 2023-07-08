import json
import os
import cv2
import pickle
import random
import argparse


def extract_bb(data, data_root, train_img_ids, val_img_ids, save_dir):
    obj_att_labels = {
        "box_index": [],
        "img_ins_files": [],
        "obj_labels": [],
        "att_labels": [],
    }

    train_index = {1: [], 2: [], 3: [], 4: []}
    val_index = {1: [], 2: [], 3: [], 4: []}

    box_index = 0

    for index, bb in enumerate(data["annotations"]):
        img_id = int(bb["image_id"])
        bb_cords = bb["bbox"]
        bb_area = bb["area"]
        img_name = [
            item["file_name"] for item in data["images"] if item["id"] == img_id
        ]
        img_path = os.path.join(data_root, img_name[0])
        img = cv2.imread(img_path)  # complete this
        x, y, w, h = bb_cords
        x, y, w, h = int(x), int(y), int(w), int(h)
        instance_id = bb["id"]

        if bb_area >= 50.0:
            bb_img = img[y : y + h, x : x + w]  # check this.
            bb_att_cls = bb["att_vec"]
            bb_obj_cls = bb["category_id"]

            out_name = os.path.join(
                save_dir,
                "bb_images/{}_id{}_n{}.png".format(
                    str(img_id).zfill(5),
                    str(instance_id).zfill(5),
                    str(box_index).zfill(5),
                ),
            )
            cv2.imwrite(out_name, bb_img)

            obj_att_labels["box_index"].append(box_index)
            obj_att_labels["img_ins_files"].append(out_name)
            obj_att_labels["obj_labels"].append(bb_obj_cls)
            obj_att_labels["att_labels"].append(bb_att_cls)

            for key, val in val_img_ids.items():
                if img_id in val:
                    val_index[key].append(box_index)
                else:
                    train_index[key].append(box_index)

            box_index += 1

        if index % 50 == 0:
            print("Annotation box-{}, ann-{}".format(box_index, index))
            print(
                "--- Train",
                len(train_index[1]),
                len(train_index[2]),
                len(train_index[3]),
                len(train_index[4]),
            )
            print(
                "--- Valid",
                len(val_index[1]),
                len(val_index[2]),
                len(val_index[3]),
                len(val_index[4]),
            )

    with open(os.path.join(save_dir, "bb_labels/ovad_labels.pkl"), "wb") as t:
        pickle.dump(obj_att_labels, t)
    with open(os.path.join(save_dir, "bb_labels/ovad_4fold_ids.pkl"), "wb") as v:
        pickle.dump(
            {
                "train_img_ids": train_img_ids,
                "train_box_ids": train_index,
                "val_img_ids": val_img_ids,
                "val_box_ids": val_index,
            },
            v,
        )

    print("Saved images and labels for box instances")
    print("Number of instances extracted {}".format(len(obj_att_labels["box_index"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-a",
        "--annotations_file",
        help="Path to annotations file in coco format",
        type=str,
        default="datasets/ovad/ovad2000.json",
    )
    parser.add_argument(
        "-d",
        "--data_root",
        help="Path to images",
        type=str,
        default="datasets/coco/val2017",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        help="Path to store new images",
        type=str,
        default="datasets/ovad_box_instances",
    )
    parser.add_argument(
        "-n",
        "--dataset_name",
        help="Name to store new json file",
        type=str,
        default="2000_img",
    )
    parser.add_argument(
        "--seed",
        help="Seed for reproduce results",
        type=int,
        default=8746,
    )
    args = parser.parse_args()

    f = open(args.annotations_file)
    data = json.load(f)

    initial_val_set = set()
    # Sets for the supervised 4-fold experiment
    train_img_ids = {1: set(), 2: set(), 3: set(), 4: set()}
    val_img_ids = {1: set(), 2: set(), 3: set(), 4: set()}

    imgId2img = {}
    for img in data["images"]:
        imgId2img[img["id"]] = img
    print("Number of images {}".format(len(data["images"])))

    all_img_ids = imgId2img.keys()
    remaining_img_ids = list(set(all_img_ids).difference(initial_val_set))
    random.shuffle(remaining_img_ids)

    all_ids_set_order = list(initial_val_set) + remaining_img_ids
    size_fold = len(all_ids_set_order) // 4
    for idx in val_img_ids.keys():
        val_ids = all_ids_set_order[(idx - 1) * size_fold : idx * size_fold]
        train_ids = set(all_ids_set_order).difference(set(val_ids))
        val_img_ids[idx] = set(val_ids)
        train_img_ids[idx] = set(train_ids)

    save_dir = os.path.join(args.save_dir, args.dataset_name)
    os.makedirs(os.path.join(save_dir, "bb_images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "bb_labels"), exist_ok=True)
    extract_bb(data, args.data_root, train_img_ids, val_img_ids, save_dir)

    print("Done")


if __name__ == "__main__":
    main()

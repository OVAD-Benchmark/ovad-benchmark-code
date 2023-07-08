"""
Copyright (c) 2022
All rights reserved.
For full license text see https://ovad-benchmark.github.io/
By Maria A. Bravo

This file saves the features for the attribute classes 
"""

import dill as pickle
import torch
import argparse
import json
import torch
import numpy as np
import itertools
import sys, os
import string
from nltk.corpus import wordnet
import ipdb

sys.path.insert(0, os.getcwd())

# get text representations
object_attribute_templates = {
    "has": {
        "none": ["{attr} {dobj} {noun}"],
        "a": ["a {attr} {dobj} {noun}", "a {noun} has {attr} {dobj}"],
        "the": ["the {attr} {dobj} {noun}", "the {noun} has {attr} {dobj}"],
        "photo": [
            "a photo of a {attr} {dobj} {noun}",
            "a photo of an {noun} which has {attr} {dobj}",
            "a photo of the {attr} {dobj} {noun}",
            "a photo of the {noun} which has {attr} {dobj}",
        ],
    },
    "is": {
        "none": ["{attr} {noun}"],
        "a": ["a {attr} {noun}", "a {noun} is {attr}"],
        "the": ["the {attr} {noun}", "the {noun} is {attr}"],
        "photo": [
            "a photo of a {attr} {noun}",
            "a photo of a {noun} which is {attr}",
            "a photo of the {attr} {noun}",
            "a photo of the {noun} which is {attr}",
        ],
    },
}


def clean_text(texts, fix_space=None, use_underscore=None):
    rm_punct = str.maketrans("", "", string.punctuation.replace("_", ""))
    if isinstance(texts, str):
        texts = [texts]
    assert isinstance(texts, list)
    clean_texts = []
    for text in texts:
        new_text = text.lower().translate(rm_punct).strip()
        if fix_space:
            new_text = new_text.replace("_", " ")
        if use_underscore:
            new_text = new_text.replace(" ", "_")
        clean_texts.append(new_text)
    clean_texts = list(set(clean_texts))
    clean_texts.sort()
    return clean_texts


def encode_clip(text_list, clip_model):
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading CLIP")
    model, preprocess = clip.load(clip_model, device=device)
    if isinstance(text_list[0], list):
        # it is a list of list of strings
        avg_synonyms = True
        sentences = list(itertools.chain.from_iterable(text_list))
        print("flattened_sentences", len(sentences))
    elif isinstance(text_list[0], str):
        avg_synonyms = False
        sentences = text_list
    text = clip.tokenize(sentences).to(device)
    with torch.no_grad():
        if len(text) > 10000:
            text_features = torch.cat(
                [
                    model.encode_text(text[: len(text) // 2]),
                    model.encode_text(text[len(text) // 2 :]),
                ],
                dim=0,
            )
        else:
            text_features = model.encode_text(text)
    print("text_features.shape", text_features.shape)
    if avg_synonyms:
        synonyms_per_cat = [len(x) for x in text_list]
        text_features = text_features.split(synonyms_per_cat, dim=0)
        text_features = [x.mean(dim=0) for x in text_features]
        text_features = torch.stack(text_features, dim=0)
        print("after stack", text_features.shape)
    text_features = text_features.cpu().numpy()
    return text_features


def encode_transformer(text_list, model, config):
    from ovadb.modeling.text.transformer_encoder import (
        PRETRAINED_MODELS,
        build_text_encoder,
    )
    from detectron2.config import get_cfg
    from ovadb.config import add_ovadb_config

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = get_cfg()
    add_ovadb_config(cfg)
    cfg.merge_from_file(config)
    model = build_text_encoder(model, cfg).to(device)
    model.eval()

    if isinstance(text_list[0], list):
        # it is a list of list of strings
        avg_synonyms = True
        sentences = list(itertools.chain.from_iterable(text_list))
        print("flattened_sentences", len(sentences))
    elif isinstance(text_list[0], str):
        avg_synonyms = False
        sentences = text_list
    with torch.no_grad():
        outputs = model(sentences)
    text_features = outputs.detach().cpu()

    if avg_synonyms:
        synonyms_per_cat = [len(x) for x in text_list]
        text_features = text_features.split(synonyms_per_cat, dim=0)
        text_features = [x.mean(dim=0) for x in text_features]
        text_features = torch.stack(text_features, dim=0)
        print("after stack", text_features.shape)
    text_features = text_features.numpy()
    print("text_features.shape", text_features.shape)
    return text_features


if __name__ == "__main__":
    """
    usage:
    python tools/dump_attribute_features.py --out_dir datasets/text_representations \
        --save_obj_categories --save_att_categories \
        --fix_space --prompt none --avg_synonyms --not_use_object --prompt_att none 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", default="datasets/ovad/ovad2000.json")
    parser.add_argument("--out_dir", default="datasets/text_representations")
    parser.add_argument("--model", default="clip")
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--config", default="")

    parser.add_argument("--save_obj_categories", action="store_true")
    parser.add_argument("--save_att_categories", action="store_true")
    parser.add_argument("--prompt", default="a")
    parser.add_argument("--fix_space", action="store_true")
    parser.add_argument("--use_underscore", action="store_true")
    parser.add_argument("--use_synonyms", action="store_true")
    parser.add_argument("--avg_synonyms", action="store_true")
    parser.add_argument("--prompt_att", default="a")
    parser.add_argument("--not_use_object", action="store_true")
    parser.add_argument("--use_wordnet", action="store_true")

    args = parser.parse_args()

    print("Loading", args.ann)
    data = json.load(open(args.ann, "r"))

    model_name = (
        "{}-{}".format(args.model, args.clip_model.replace("/", ""))
        if "clip" in args.model
        else args.model
    )

    if args.save_obj_categories or not args.not_use_object:
        orig_cat_names = [
            x["name"] for x in sorted(data["categories"], key=lambda x: x["id"])
        ]
        cat_names = orig_cat_names
        # Get object categories' synonyms
        if args.use_synonyms:
            from ovadb.data.datasets.utils import coco_to_synset
            from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES

            synset_to_synonyms = {x["synset"]: x["synonyms"] for x in LVIS_CATEGORIES}
            synonyms = [
                synset_to_synonyms[coco_to_synset[x["name"]]["synset"]]
                if coco_to_synset[x["name"]]["synset"] in synset_to_synonyms.keys()
                else [x["name"]]
                for x in data["categories"]
            ]

            if args.use_wordnet:
                synonyms_wn = [
                    [
                        xx.name()
                        for xx in wordnet.synset(
                            coco_to_synset[x["name"]]["synset"]
                        ).lemmas()
                    ]
                    if coco_to_synset[x["name"]]["synset"] != "stop_sign.n.01"
                    else [x["name"]]
                    for x in sorted(data["categories"], key=lambda x: x["id"])
                ]
                new_syn = [
                    syn_lvis + syn_wn for syn_lvis, syn_wn in zip(synonyms, synonyms_wn)
                ]
                synonyms = new_syn

            # remove punctuation and clean text
            clean_cat = []
            clean_syn = []
            for cat_idx in range(len(cat_names)):
                clean_cat.append(
                    clean_text(cat_names[cat_idx], args.fix_space, args.use_underscore)[
                        0
                    ]
                )
                if args.use_synonyms:
                    clean_syn.append(
                        clean_text(
                            synonyms[cat_idx], args.fix_space, args.use_underscore
                        )
                    )
            cat_names = clean_cat
            synonyms = clean_syn
            print("cat_names", cat_names)
        else:
            synonyms = [[cat] for cat in cat_names]

        if args.save_obj_categories:
            # get the prompted text
            if args.prompt == "a":
                sentences = ["a " + x for x in cat_names]
                sentences_synonyms = [["a " + xx for xx in x] for x in synonyms]
            elif args.prompt == "the":
                sentences = ["the " + x for x in cat_names]
                sentences_synonyms = [["the " + xx for xx in x] for x in synonyms]
            elif args.prompt == "none":
                sentences = [x for x in cat_names]
                sentences_synonyms = [[xx for xx in x] for x in synonyms]
            elif args.prompt == "photo":
                sentences = ["a photo of a {}".format(x) for x in cat_names]
                sentences_synonyms = [
                    ["a photo of a {}".format(xx) for xx in x] for x in synonyms
                ]
            elif args.prompt == "scene":
                sentences = [
                    "a photo of a {} in the scene".format(x) for x in cat_names
                ]
                sentences_synonyms = [
                    ["a photo of a {} in the scene".format(xx) for xx in x]
                    for x in synonyms
                ]
            else:  # use a combination "a", "the" and "none"
                sentences = ["a " + x for x in cat_names]
                sentences_synonyms = []
                for x in synonyms:
                    sub_sentences_synonyms = []
                    for prompt in ["a ", "the ", ""]:
                        sub_sentences_synonyms += [prompt + xx for xx in x]
                    sentences_synonyms.append(sub_sentences_synonyms)
            print(
                "sentences_synonyms",
                len(sentences_synonyms),
                sum(len(x) for x in sentences_synonyms),
            )

            # get features
            if args.model == "clip":
                if args.avg_synonyms:
                    text_features = encode_clip(sentences_synonyms, args.clip_model)
                else:
                    text_features = encode_clip(sentences, args.clip_model)
            elif args.model in ["bert", "roberta"]:
                if args.avg_synonyms:
                    text_features = encode_transformer(
                        sentences_synonyms, args.model, args.config
                    )
                else:
                    text_features = encode_transformer(
                        sentences, args.model, args.config
                    )
            else:
                assert 0, args.model

            # save noun embeddings
            if args.out_dir != "":
                out_path = os.path.join(
                    args.out_dir,
                    "ovad_obj{}_{}_{}+cname.npy".format(
                        "_syn" if args.use_synonyms else "_cls",
                        args.model
                        + (
                            args.clip_model.replace("/", "")
                            if args.model == "clip"
                            else ""
                        ),
                        args.prompt,
                    ),
                )

                save_dict = {}
                for cat, feat in zip(orig_cat_names, text_features):
                    save_dict[cat] = feat

                print("saving to", out_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(open(out_path, "wb"), text_features)

                file_name = out_path.replace(".npy", ".pkl")
                with open(file_name, "wb") as handle:
                    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.save_att_categories:
        """
        Second part. Attribute embeddings
        """
        # attribute embeddings
        attributes_data = data["attributes"]
        """
        {
            "id": 0,
            "name": "cleanliness:clean/neat",
            "type": "cleanliness",
            "parent_type": "cleanliness",
            "is_has_att": "is",
            "freq_set": "head"
        },
        """
        templates_dict = object_attribute_templates

        if args.prompt_att in templates_dict["is"].keys():
            use_prompts = [args.prompt_att]
        else:  # use all prompts
            use_prompts = ["a", "the", "none"]

        if args.not_use_object:
            object_word = ""
        else:
            object_word = "object"

        # unconditional embeddings
        all_att_templates = []
        for att_dict in attributes_data:
            att_w_type = att_dict["name"]
            att_type, att_list = att_w_type.split(":")
            assert att_type == att_dict["type"]
            is_has = att_dict["is_has_att"]

            dobj_name = (
                att_type.replace(" tone", "")
                # So far only for tone worked to remove the word
                # .replace(" color", "")
                # .replace(" pattern", "")
                # .replace(" expression", "")
                # .replace(" type", "")
                # .replace(" length", "")
            )

            # extend the maturity to include other words
            if att_list == "young/baby":
                att_list += "/kid/kids/child/toddler/boy/girl"
            elif att_list == "adult/old/aged":
                att_list += "/teen/elder"

            att_templates = []
            for syn in att_list.split("/"):
                for prompt in use_prompts:
                    for template in templates_dict[is_has][prompt]:
                        if is_has == "has":
                            att_templates.append(
                                template.format(
                                    attr=syn, dobj=dobj_name, noun=object_word
                                ).strip()
                            )
                        elif is_has == "is":
                            att_templates.append(
                                template.format(attr=syn, noun=object_word).strip()
                            )
            all_att_templates.append(att_templates)

        if not args.avg_synonyms:
            att_templates_syn = all_att_templates
            len_synonyms = [len(att_synonyms) for att_synonyms in all_att_templates]
            att_ids = [
                [att_dict["id"]] * len(att_synonyms)
                for att_dict, att_synonyms in zip(attributes_data, att_templates_syn)
            ]
            att_ids = list(itertools.chain.from_iterable(att_ids))
            all_att_templates = list(itertools.chain.from_iterable(all_att_templates))

        if args.model == "clip":
            text_features = encode_clip(all_att_templates, args.clip_model)
        elif args.model in ["bert", "roberta"]:
            text_features = encode_transformer(
                all_att_templates, args.model, args.config
            )
        else:
            assert 0, args.model
        print("Text for unconditional att", all_att_templates)

        # save att uncoditional embeddings
        if args.out_dir != "":
            if args.avg_synonyms:
                out_path = os.path.join(
                    args.out_dir,
                    "ovad_att_{}_{}+catt.npy".format(
                        model_name,
                        (args.prompt_att if args.prompt_att != "none" else "")
                        + object_word,
                    ),
                )

                print("saving to", out_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(open(out_path, "wb"), text_features)

                save_dict = {}
                for cat, feat in zip(attributes_data, text_features):
                    save_dict[cat["name"]] = feat

                file_name = out_path.replace(".npy", ".pkl")
                with open(file_name, "wb") as handle:
                    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if not args.avg_synonyms:
                out_path = os.path.join(
                    args.out_dir,
                    "ovad_att_{}_{}+catt_syn.pkl".format(
                        model_name,
                        (args.prompt_att if args.prompt_att != "none" else "")
                        + object_word,
                    ),
                )
                text_prompt_dict = {
                    "ids": att_ids,
                    "syn_len": len_synonyms,
                    "feat": text_features,
                    "syn_text": all_att_templates,
                }
                print("saving to", out_path)
                with open(out_path, "wb") as handle:
                    pickle.dump(
                        text_prompt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL
                    )

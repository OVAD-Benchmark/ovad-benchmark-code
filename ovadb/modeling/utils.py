# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import json
import pickle
import numpy as np
from torch import nn
from torch.nn import functional as F

from detectron2.data import MetadataCatalog


def load_class_freq(
    path="datasets/metadata/lvis_v1_train_cat_info.json", freq_weight=1.0
):
    cat_info = json.load(open(path, "r"))
    cat_info = torch.tensor(
        [c["image_count"] for c in sorted(cat_info, key=lambda x: x["id"])]
    )
    freq_weight = cat_info.float() ** freq_weight
    return freq_weight


def get_fed_loss_inds(gt_classes, num_sample_cats, C, weight=None):
    appeared = torch.unique(gt_classes)  # C'
    prob = appeared.new_ones(C + 1).float()
    prob[-1] = 0
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:C] = weight.float().clone()
        prob[appeared] = 0
        more_appeared = torch.multinomial(
            prob, num_sample_cats - len(appeared), replacement=False
        )
        appeared = torch.cat([appeared, more_appeared])
    return appeared


def reset_cls_test(model, cls_path, num_classes, dataset_name):
    """
    loads and changes the class object embeddings for the test classes
    """
    # gets model if it is in multiple gpus
    module = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    # ensures it has a roi_heads module
    assert hasattr(module, "roi_heads")
    module.roi_heads.num_classes = num_classes

    # loads the class weights
    if type(cls_path) == str and cls_path.endswith(".npy"):
        print("Resetting zs_weight", cls_path)
        zs_weight = (
            torch.tensor(np.load(cls_path), dtype=torch.float32)
            .permute(1, 0)
            .contiguous()
        )  # D x C
    elif type(cls_path) == str and cls_path.endswith(".pkl"):
        print("Resetting zs_weight", cls_path)
        weight_dict = pickle.load(open(cls_path, "rb"))

        metadata = MetadataCatalog.get(dataset_name)
        thing_classes = metadata.thing_classes

        assert len(thing_classes) == num_classes

        zs_weight = []
        for cls in thing_classes:
            zs_weight.append(np.asarray(weight_dict[cls]))
        zs_weight = (
            torch.tensor(np.asarray(zs_weight), dtype=torch.float32)
            .permute(1, 0)
            .contiguous()
        )
    else:
        zs_weight = cls_path

    # verifies that the number of classes matches the size of the weights
    assert (
        zs_weight.shape[1] == num_classes
    ), "Number of classes loaded in weights {} must match classes in metadata {}".format(
        zs_weight.shape[1], num_classes
    )
    # adds background embedding
    zs_weight = torch.cat(
        [zs_weight, zs_weight.new_zeros((zs_weight.shape[0], 1))], dim=1
    )  # D x (C + 1)

    # Modifies module in network
    if isinstance(module.roi_heads.box_predictor, nn.ModuleList):
        for box_pdr in module.roi_heads.box_predictor:
            if box_pdr.cls_score.norm_weight:
                zs_weight_pd = F.normalize(zs_weight, p=2, dim=0)
            zs_weight_pd = zs_weight_pd.to(module.device)

            del box_pdr.cls_score.zs_weight
            box_pdr.cls_score.zs_weight = zs_weight_pd
    elif isinstance(module.roi_heads.box_predictor, nn.Module):
        if module.roi_heads.box_predictor.cls_score.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)
        zs_weight = zs_weight.to(module.device)

        del module.roi_heads.box_predictor.cls_score.zs_weight
        module.roi_heads.box_predictor.cls_score.zs_weight = zs_weight
    else:
        assert False, "The network could not load the obj embeddings {}".format(
            module.roi_heads.box_predictor
        )


def reset_attribute_test_model(
    model, dataset_name, noun_weights_path, att_weights_path
):
    """
    loads and changes the class attribute embeddings for the test classes
    """
    module = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    assert hasattr(module, "roi_heads")
    metadata = MetadataCatalog.get(dataset_name)

    # load nouns
    num_classes = len(metadata.thing_classes)
    if type(noun_weights_path) == str and noun_weights_path.endswith(".npy"):
        print("Resetting zs_weight", noun_weights_path)
        zs_weight = torch.tensor(
            np.load(noun_weights_path), dtype=torch.float32
        )  # C x D
    elif type(noun_weights_path) == str and noun_weights_path.endswith(".pkl"):
        print("Resetting zs_weight", noun_weights_path)
        weight_dict = pickle.load(open(noun_weights_path, "rb"))

        metadata = MetadataCatalog.get(dataset_name)
        thing_classes = metadata.thing_classes
        assert len(thing_classes) == num_classes

        zs_weight = []
        for cls in thing_classes:
            zs_weight.append(np.asarray(weight_dict[cls]))
        zs_weight = torch.tensor(np.asarray(zs_weight), dtype=torch.float32)
    else:
        zs_weight = noun_weights_path

    assert (
        zs_weight.shape[0] == num_classes
    ), "Number of classes loaded in weights {} must match classes in metadata {}".format(
        zs_weight.shape[0], num_classes
    )

    module.roi_heads.attribute_head.set_embeddings(
        "", is_noun=True, zs_weight=zs_weight
    )
    assert (
        module.roi_heads.attribute_head.num_classes == num_classes
    ), "Number of classes loaded in weights {} must match classes in metadata {}".format(
        module.roi_heads.attribute_head.num_classes, num_classes
    )

    # load attributes
    num_attributes = len(metadata.attribute_classes)
    module.roi_heads.attribute_head.set_embeddings(att_weights_path, is_noun=False)
    assert (
        module.roi_heads.attribute_head.num_attributes == num_attributes
    ), "Number of attributes loaded in weights {} must match classes in metadata {}".format(
        module.roi_heads.attribute_head.num_attributes, num_attributes
    )

    module.roi_heads.attribute_on = True


def restore_attribute_test_model(model):
    module = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    assert hasattr(module, "roi_heads")
    module.roi_heads.attribute_on = False

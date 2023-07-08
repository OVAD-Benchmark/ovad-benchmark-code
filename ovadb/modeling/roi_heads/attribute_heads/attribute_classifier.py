import os
import numpy as np
import torch
from torch import nn
import dill as pickle
from torch.nn import functional as F
from detectron2.config import configurable
from typing import Dict, List, Optional, Tuple
from detectron2.layers import Linear, ShapeSpec
from detectron2.structures import Boxes, Instances

from ovadb.modeling.logged_module import LoggedModule

"""
Similar to ZeroShotClassifier but including atttributes
"""


class AttributeClassifier(LoggedModule):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        zs_weight_path: str,
        num_attributes: int,
        att_weight_path: str,
        zs_weight_dim: int = 512,
        use_bias: float = 0.0,
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
        norm_temp_att: float = 50.0,
        use_sigmoid_ce: bool = False,
        add_feature: bool = False,
        conditional_prediction: bool = False,
        conditional_weights: str = "",
        conditional_obj_label: bool = False,
        norm_temp_cond_att: float = 50.0,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = (
            input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        )
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature
        self.norm_temp_att = norm_temp_att
        self.zs_weight_dim = zs_weight_dim
        self.use_sigmoid_ce = use_sigmoid_ce
        self.use_bias = use_bias < 0
        self.bias = use_bias

        # Add the embedding based layer
        # nouns
        self.num_classes = num_classes
        self.noun_pred = nn.Linear(self.zs_weight_dim, self.num_classes)
        nn.init.normal_(self.noun_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.noun_pred.bias, 0)

        zs_weight = torch.randn((num_classes, zs_weight_dim))
        nn.init.normal_(zs_weight, std=0.01)
        self.set_embeddings(zs_weight_path, is_noun=True, zs_weight=zs_weight)

        # attributes
        self.num_attributes = num_attributes
        self.attr_pred = nn.Linear(self.zs_weight_dim, self.num_attributes)
        nn.init.normal_(self.attr_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.attr_pred.bias, 0)
        self.att_syn_len = [1] * num_attributes
        self.att_ids = list(range(num_attributes))

        zs_weight = torch.randn((num_attributes, zs_weight_dim))
        nn.init.normal_(zs_weight, std=0.01)
        self.set_embeddings(att_weight_path, is_noun=False, zs_weight=zs_weight)

        self.add_feature = add_feature

        self.conditional_prediction = conditional_prediction
        if conditional_prediction:
            self.set_conditional_embeddings(
                conditional_weights, num_classes, num_attributes
            )
            self.use_label = conditional_obj_label
            self.norm_temp_cond_att = norm_temp_cond_att

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "zs_weight_path": cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            "zs_weight_dim": cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            "use_bias": cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            "norm_weight": cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            "norm_temperature": cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
            "norm_temp_att": cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP_ATTRIBUTE,
            "num_attributes": cfg.MODEL.ROI_HEADS.NUM_ATTRIBUTES,
            "att_weight_path": cfg.MODEL.ROI_BOX_HEAD.ATTRIBUTE_WEIGHT_PATH,
            "use_sigmoid_ce": cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE,
            "add_feature": cfg.MODEL.ROI_BOX_HEAD.ADD_BOX_FEATURES_PREDICTION,
            "conditional_prediction": cfg.EVALUATION_ATTRIBUTE.CONDITIONAL,
            "conditional_weights": cfg.EVALUATION_ATTRIBUTE.EMBEDDING_DICTIONARY,
            "conditional_obj_label": cfg.EVALUATION_ATTRIBUTE.CONDITIONAL_USE_OBJ_LABEL,
            "norm_temp_cond_att": cfg.EVALUATION_ATTRIBUTE.CONDITIONAL_TEMPERATURE,
        }

    def set_embeddings(self, path_weights, is_noun=True, zs_weight=None):
        assert (
            os.path.isfile(path_weights) or zs_weight is not None
        ), "Path to classification weights must be valid: {}".format(path_weights)

        # get weights
        device = self.noun_pred.weight.device
        if os.path.isfile(path_weights):
            print("Loading {} for attribute head".format(path_weights))
            # if saved as numpy - synonyms are average
            if path_weights.endswith(".npy"):
                zs_weight = torch.tensor(
                    np.load(path_weights), dtype=torch.float32
                )  # C x D
                self.att_ids = list(range(zs_weight.shape[0]))
                self.att_syn_len = [1] * zs_weight.shape[0]
                self.num_attributes = zs_weight.shape[0]
            # saved as pickle
            elif path_weights.endswith(".pkl"):
                att_syn_dict = pickle.load(open(path_weights, "rb"))
                self.att_syn_len = att_syn_dict["syn_len"]
                self.att_ids = att_syn_dict["ids"]
                self.num_attributes = len(self.att_syn_len)
                zs_weight = torch.tensor(
                    att_syn_dict["feat"], dtype=torch.float32
                )  # C x D
        if torch.is_tensor(zs_weight):
            zs_weight = zs_weight.clone().detach().to(device)
        else:
            zs_weight = torch.tensor(zs_weight, device=device)

        assert (
            zs_weight.shape[1] == self.zs_weight_dim
        ), "The weigts dimension {} has to match the one saved in the model {}".format(
            zs_weight.shape[1], self.zs_weight_dim
        )

        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=1)

        self.log("zs_weight", zs_weight)

        # noun
        if is_noun:
            self.num_classes = zs_weight.shape[0]
            zs_weight = torch.cat(
                [zs_weight, zs_weight.new_zeros((1, self.zs_weight_dim))], dim=0
            )  # (C + 1) x D
            self.noun_pred = nn.Linear(self.zs_weight_dim, self.num_classes + 1)
            self.noun_pred.weight.data = zs_weight
            if self.use_bias:
                self.noun_pred.bias.data = (
                    torch.ones_like(self.noun_pred.bias.data, device=device) * self.bias
                )
            else:
                self.noun_pred.bias.data = torch.zeros_like(
                    self.noun_pred.bias.data, device=device
                )

        # attr
        if not is_noun:
            self.attr_pred = nn.Linear(self.zs_weight_dim, zs_weight.shape[0])
            self.attr_pred.weight.data = zs_weight
            if self.use_bias:
                self.attr_pred.bias.data = (
                    torch.ones_like(self.attr_pred.bias.data, device=device) * self.bias
                )
            else:
                self.attr_pred.bias.data = torch.zeros_like(
                    self.attr_pred.bias.data, device=device
                )

        self.freeze_classifiers(is_noun)

    def freeze_classifiers(self, is_noun=True):
        if is_noun:
            self.noun_pred.weight.requires_grad = False
            self.noun_pred.bias.requires_grad = False
        if not is_noun:
            self.attr_pred.weight.requires_grad = False
            self.attr_pred.bias.requires_grad = False

    def set_conditional_embeddings(
        self,
        conditional_weights_path,
        num_classes,
        num_attributes,
        idCls2cls=None,
        idAtt2att=None,
    ):
        assert os.path.isfile(
            conditional_weights_path
        ), "Path to classification weights must be valid: {}".format(
            conditional_weights_path
        )
        self.dict_noun_att = pickle.load(open(conditional_weights_path, "rb"))
        if idCls2cls is not None:
            self.idCls2cls = idCls2cls
        else:
            self.idCls2cls = {}
        for key, val in self.dict_noun_att.items():
            zs_weight = torch.tensor(np.asarray(val), dtype=torch.float32)
            if self.norm_weight:
                zs_weight = F.normalize(zs_weight, p=2, dim=1)
            self.dict_noun_att[key] = zs_weight
            if key not in self.idCls2cls.keys():
                self.idCls2cls[len(self.idCls2cls)] = key

    def forward(
        self, x, instances: List[Instances], classifier=None, attribute_cls=None
    ):
        """
        Inputs:
            x: N x D
            per-region features of shape (N, ...) for N bounding boxes to predict.
            classifier: (Cn x D)
            attribute_cls: (Ca x D)
        """

        if classifier is not None:
            self.set_embeddings("", is_noun=True, zs_weight=classifier)
        if attribute_cls is not None:
            self.set_embeddings("", is_noun=False, zs_weight=attribute_cls)

        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        if self.norm_weight:
            x_noun = self.norm_temperature * F.normalize(x, p=2, dim=1)
            x_attr = self.norm_temp_att * F.normalize(x, p=2, dim=1)
        else:
            x_noun = x
            x_attr = x

        x_nouns = self.noun_pred(x_noun)
        x_attrs = self.attr_pred(x_attr)

        # take max over att synonyms
        if self.num_attributes != len(self.att_ids):
            # split into synonyms
            x_attrs_syn = x_attrs.split(self.att_syn_len, dim=1)
            # take arg max
            x_attrs_maxsyn = []
            x_attrs_idxsyn = []
            for x_syn in x_attrs_syn:
                xmax_val, xmax_idx = x_syn.max(axis=1)
                x_attrs_maxsyn.append(xmax_val)
                x_attrs_idxsyn.append(xmax_idx)
            x_attrs = torch.stack(x_attrs_maxsyn, axis=1)

        instances = self.attribute_rcnn_inference(x, x_nouns, x_attrs, instances)

        if self.add_feature:
            num_inst_per_image = [len(p) for p in instances]
            features = x.split(num_inst_per_image, dim=0)
            for feature, instance in zip(features, instances):
                instance.features = feature

        return instances

    def predict_probs(self, scores, proposals):
        """
        support sigmoid
        """
        num_inst_per_image = [len(p) for p in proposals]
        if self.use_sigmoid_ce:
            probs = scores.sigmoid()
        else:
            probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

    def predict_att_prob(self, scores, proposals):
        """
        applies sigmoid
        """
        num_inst_per_image = [len(p) for p in proposals]
        probs = scores.sigmoid()
        return probs.split(num_inst_per_image, dim=0)

    def pred_conditional_prob(self, x, instance):
        if self.norm_weight:
            x = self.norm_temp_cond_att * F.normalize(x, p=2, dim=1)

        x_cond_attrs = []
        for box_idx in range(len(instance)):
            # get index of object
            if self.use_label:
                # TODO: get labels until this point
                assert instance.has(
                    "gt_classes"
                ), "Instance does not have gt_classes to do conditional prediction"
                noun_idx = instance.gt_classes[box_idx].item()
            else:
                # use predicted labels
                noun_idx = instance.pred_classes[box_idx].item()

            atts_vector = self.dict_noun_att[self.idCls2cls[noun_idx]].to(x.device)
            x_cond_attr = torch.mm(x[box_idx : box_idx + 1], atts_vector.T)
            x_cond_attrs.append(x_cond_attr)

        if len(x_cond_attrs) > 0:
            x_cond_attrs = torch.cat(x_cond_attrs, axis=0)

            if self.use_bias:
                x_cond_attrs = x_cond_attrs + self.cls_bias

            cond_attr_prob = self.predict_att_prob(x_cond_attrs, [[0] * len(x)])[0]
            instance.cond_att_scores = cond_attr_prob
        else:
            instance.cond_att_scores = torch.zeros(0).to(self.noun_pred.weight.device)

    def attribute_rcnn_inference(
        self,
        x,
        x_nouns: torch.Tensor,
        x_attrs: torch.Tensor,
        instances: List[Instances],
    ):
        noun_prob = self.predict_probs(x_nouns, instances)
        attr_prob = self.predict_att_prob(x_attrs, instances)

        for noun, att, instance in zip(noun_prob, attr_prob, instances):
            instance.noun_scores = noun
            instance.att_scores = att

            if self.conditional_prediction:
                self.pred_conditional_prob(x, instance)

        return instances

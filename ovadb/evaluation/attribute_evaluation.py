import os
import io
import copy
import json
import torch
import logging
import itertools
import contextlib
import numpy as np
from tabulate import tabulate
from pickletools import uint8
from pycocotools.coco import COCO
from collections import OrderedDict
from torch.nn import functional as F
import pycocotools.mask as mask_util

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco

from datasets.ovad.ovad_evaluator import AttEvaluator

from ovadb.data.datasets.coco_ovd import categories_base
from ovadb.data.datasets.ovad import convert_to_coco_extended_json


def instances_to_coco_json_plus_fields(instances, img_id, fields=[]):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    other_fields = {}
    for name_field in fields:
        if instances.has(name_field):
            val_field = instances.get(name_field)
            other_fields[name_field] = val_field.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        for f, v in other_fields.items():
            result[f] = v[k]
        results.append(result)
    return results


# TODO: WRITE AND SAVE THESE VECTORS IN METADATA
def apply_attribute_prediction(
    pred_class, pred_attributes, attribute_hierarchy, att2idx
):
    mask_att_ignore = np.ones_like(pred_attributes)
    new_att_score = torch.tensor(pred_attributes)
    # for att_type, att in attribute_hierarchy['hierarchy']['human']:

    if pred_class == "person":
        att_hierarchy = attribute_hierarchy["hierarchy"]["human"]
    elif pred_class in {
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    }:
        att_hierarchy = attribute_hierarchy["hierarchy"]["animal"]
    elif pred_class in {
        "food-other-merged",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "fruit",
    }:
        att_hierarchy = attribute_hierarchy["hierarchy"]["food"]
    else:
        att_hierarchy = attribute_hierarchy["hierarchy"]["object"]

    # make distributions of prob across attributes which are mutually exclusive
    for att_type, att in att_hierarchy.items():
        softmax_idx = []
        for ex_att in att:
            if att_type == "material":
                ex_att = "/".join(attribute_hierarchy["extended_materials"][ex_att])
            att_name = att_type + ":" + ex_att
            softmax_idx.append(att2idx[att_name])
        mask_att_ignore[softmax_idx] = 0
        new_att_score[softmax_idx] = F.softmax(new_att_score[softmax_idx], dim=-1)
    new_att_score[mask_att_ignore == 1] = 0

    return new_att_score


# inspired from COCOEval extended to multiclass
def evaluate_attribute_detection(
    predictions,
    attributes,
    metadata,
    output_dir,
    metrics=None,
    return_gt_pred_pair=False,
    cls_depend=False,
    min_iou_match=0.5,
    include_missing_boxes=True,
    conditional=False,
):
    """
    Evaluate attribute detection metrics.
    """
    # 1. match gt bounding box with best predicted bounding box
    # 2. get attributes for every bounding box matched
    gt_overlaps = []
    gt_pred_match = []
    no_gt = 0
    no_pred = 0
    no_corrs = 0
    gt_vec = []
    pred_vec = []
    att_key = "cond_att_scores" if conditional else "att_scores"
    imgId2attBoxes = {sample["image_id"]: sample for sample in attributes}
    logger = logging.getLogger(__name__)

    for img_prediction in predictions:
        img_id = img_prediction["image_id"]
        prediction = img_prediction["instances"]

        # Discard images without predictions
        if len(prediction) == 0:
            no_pred += 1
            continue

        # Get ground truth labels
        ground_truth = imgId2attBoxes[img_id]

        # Discard images without labels
        if len(ground_truth["id_instances"]) == 0:
            no_gt += 1
            continue

        # Build Boxes with ground truth and predictions
        gt_boxes = np.asarray(ground_truth["gt_boxes"])
        # gt_boxes = BoxMode.convert(gt_boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        gt_boxes = Boxes(gt_boxes)

        pd_boxes = np.asarray([pred["bbox"] for pred in prediction])
        pd_boxes = BoxMode.convert(pd_boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        pd_boxes = Boxes(pd_boxes)

        # Calculate the IoU between ground truth and predictions
        overlaps = pairwise_iou(pd_boxes, gt_boxes)
        # Consider IoUs bigger than min_iou_match (0.5)
        overlaps[overlaps < min_iou_match] = -1

        # get best matching proposals - gt boxes
        _gt_overlaps = torch.zeros(len(gt_boxes))
        gt_prop_corresp = []
        for j in range(min(len(pd_boxes), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            if gt_ovr < 0:
                continue
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            gt_prop_corresp.append((gt_ind, box_ind, _gt_overlaps[j]))
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        if len(gt_prop_corresp) != len(gt_boxes):
            no_corrs += len(gt_boxes) - len(gt_prop_corresp)
            # print('{} No correspondance boxes id {}. gt:{}, pred:{}, match:{}'.format(no_corrs, original_id, len(gt_boxes), len(prediction), len(gt_prop_corresp)))
            # TODO: find missing gt box and report positive att as FN
            if include_missing_boxes:
                # Find missing gt indexes
                matching_gt_ind = set(
                    [match_triple[0].item() for match_triple in gt_prop_corresp]
                )
                missing_gt_id = set(range(len(gt_boxes))).difference(matching_gt_ind)
                # Include missing indexes to the predictions as if model had not predicted anything -> 0 val prediction
                for gt_ind in missing_gt_id:
                    gt_att_vec = ground_truth["gt_att_vec"][gt_ind]
                    att_scores = [0.0] * len(gt_att_vec)
                    gt_vec.append(gt_att_vec)
                    pred_vec.append(att_scores)

        for gt_ind, box_ind, iou_score in gt_prop_corresp:
            obj_pair = {"image_id": ground_truth["image_id"]}
            for key, val in ground_truth.items():
                if isinstance(val, list):
                    obj_pair[key] = val[gt_ind]
            box_pred = pd_boxes[box_ind : box_ind + 1]
            obj_pred = prediction[box_ind]
            for key in obj_pred.keys():
                if key not in obj_pair.keys():
                    obj_pair[key] = obj_pred[key]
            obj_pair["box_pred"] = box_pred
            obj_pair["iou_score"] = iou_score
            _ = obj_pair.pop("bbox")
            gt_pred_match.append(obj_pair)

            if cls_depend:
                try:
                    cls_depend_att_scores = apply_attribute_prediction(
                        metadata.thing_classes[obj_pair["category_id"]],
                        obj_pair[att_key],
                        metadata.attribute_hierarchy,
                        metadata.att2idx,
                    )
                    obj_pair[att_key] = cls_depend_att_scores.tolist()
                except:
                    print(
                        "!! Failed to do class dependent evaluation - \n"
                        + "It was not possible to have distribution across exclusive attributes"
                    )

            gt_vec.append(obj_pair["gt_att_vec"])
            pred_vec.append(np.asarray(obj_pair[att_key]))

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)

    if len(gt_overlaps) == 0:
        logger.info("No matching predictions found")
        return {}

    if return_gt_pred_pair:
        return gt_pred_match
    else:
        file_path = (
            output_dir + ("_cond" if conditional else "_uncond") + "_gt_pred_match.pth"
        )
        with PathManager.open(file_path, "wb") as f:
            torch.save(gt_pred_match, f)

    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    pred_vec = np.vstack(pred_vec)
    gt_vec = np.vstack(gt_vec)

    gt_vec[gt_vec == -1] = 2

    attribute_list = metadata.attribute_classes
    assert len(attribute_list) == pred_vec.shape[1]
    assert len(attribute_list) == gt_vec.shape[1]
    logger.info("Number of missing objects {}".format(no_corrs))

    """
    - pred:     prediction matrix [n_instance, n_class].
                pred[i,j] is the j-th attribute score of instance i-th.
                These scores should be from 0 -> 1.
    - gt_label: groundtruth label matrix [n_instances, n_class].
                gt_label[i,j] = 1 if instance i is positively labeled with
                attribute j, = 0 if it is negatively labeled, and = 2 if
                it is unlabeled.
    """

    output_file = os.path.join(
        output_dir + ("_cond" if conditional else "_uncond"),
        "{}_att.log".format(metadata.name),
    )
    threshold = float(int(pred_vec.mean() * 10) / 10)
    topk = int((gt_vec == 1).sum(1).mean())

    att_evaluator = AttEvaluator(
        metadata.att2idx,
        attr_type=metadata.att_type,
        attr_parent_type=metadata.att_parent_type,
        attr_headtail=metadata.attribute_head_tail,
        att_seen_unseen=metadata.att_base_novel,
        dataset_name=metadata.name,
        threshold=threshold,
        top_k=topk,
        exclude_atts=[],
        output_file=output_file,
    )
    att_evaluator.register_gt_pred_vectors(
        gt_vec.copy().astype("float64"),
        pred_vec.copy().astype("float64"),
        copy.deepcopy(gt_pred_match),
    )

    # results = {}
    # # Evaluate with threshold
    # scores_overall, scores_per_class = att_evaluator.evaluate(
    #     pred_vec.copy().astype('float64'),
    #     gt_vec.copy().astype('float64'),
    #     )
    # results['Thr({})'.format(threshold)] = {
    #     "Overall":scores_overall,
    #     "Per_cls":scores_per_class,
    # }
    # # Evaluate with topk (k=mean of #att per instance)
    # scores_overall_topk, scores_per_class_topk = att_evaluator.evaluate(
    #     pred_vec.copy().astype('float64'),
    #     gt_vec.copy().astype('float64'),
    #     threshold_type='topk',
    #     )
    # results['Topk({})'.format(topk)] = {
    #     "Overall":scores_overall_topk,
    #     "Per_cls":scores_per_class_topk,
    # }
    # results = att_evaluator.print_evaluation(
    #     pred_vec.copy().astype('float64'),
    #     gt_vec.copy().astype('float64'),
    #     output_file
    #     )
    # rnd_results = {}
    # for idx in range(3):
    #     rnd_results = att_evaluator.calc_rand_scores(gt_vec.copy().astype('float64'), run_n=str(idx))
    #     results.update(rnd_results)

    """
    Calculate random score to compare numbers
    """
    # random_pred = np.random.rand(*pred_vec.shape)
    # random_pred = np.ones_like(pred_vec)*0.51
    # results = att_evaluator.print_evaluation(
    #     random_pred.copy().astype('float64'),
    #     gt_vec.copy().astype('float64'),
    #     output_file
    #     )

    return att_evaluator


class AttributeEvaluator(COCOEvaluator):
    """
    Similar to COCOEvaluator with additional:
    - saves attribute labels on the format file
    - keeps labels online for evaluation
    - keeps attribute preditions
    """

    def __init__(
        self,
        dataset_name,
        cfg,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
        noun_weights_path="",
        att_weights_path="",
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                   contains all the results in the format they are produced by the model.
                2. "coco_instances_results.json" a json file in COCO's result format.
            max_dets_per_image (int): limit on the maximum number of detections per image.
                By default in COCO, this limit is to 100, but this can be customized
                to be greater, as is needed in evaluation metrics AP fixed and AP pool
                (see https://arxiv.org/pdf/2102.01066.pdf)
                This doesn't affect keypoint evaluation.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
            kpt_oks_sigmas (list[float]): The sigmas used to calculate keypoint OKS.
                See http://cocodataset.org/#keypoints-eval
                When empty, it will use the defaults in COCO.
                Otherwise it should be the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        # COCOeval requires the limit on the number of detections per image (maxDets) to be a list
        # with at least 3 elements. The default maxDets in COCOeval is [1, 10, 100], in which the
        # 3rd element (100) is used as the limit on the number of detections per image when
        # evaluating AP. COCOEvaluator expects an integer for max_dets_per_image, so for COCOeval,
        # we reformat max_dets_per_image into [1, 10, max_dets_per_image], based on the defaults.
        if max_dets_per_image is None:
            max_dets_per_image = [1, 10, 100]
        else:
            max_dets_per_image = [1, 10, max_dets_per_image]
        self._max_dets_per_image = max_dets_per_image

        if tasks is not None and isinstance(tasks, CfgNode):
            kpt_oks_sigmas = (
                tasks.TEST.KEYPOINT_OKS_SIGMAS if not kpt_oks_sigmas else kpt_oks_sigmas
            )
            self._logger.warn(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            if output_dir is None:
                raise ValueError(
                    "output_dir must be provided to COCOEvaluator "
                    "for datasets not in COCO format."
                )
            self._logger.info(f"Trying to convert '{dataset_name}' to COCO format ...")

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_extended_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset
        if self._do_evaluation:
            self._kpt_oks_sigmas = kpt_oks_sigmas

        # Load class (noun and attribute) features
        self.noun_weights_path = noun_weights_path
        self.num_classes = len(self._metadata.thing_classes)
        self.noun_weights = np.load(self.noun_weights_path)
        assert (
            self.num_classes == self.noun_weights.shape[0]
        ), "Number of classes loaded in weights {} must match classes in metadata {}".format(
            self.noun_weights.shape[0], self.num_classes
        )
        self.att_weights_path = att_weights_path
        self.num_attributes = len(self._metadata.attribute_classes)
        # self.att_weights = np.load(self.att_weights_path)
        # assert (
        #     self.num_attributes == self.att_weights.shape[0]
        # ), "Number of attributes loaded in weights {} must match attributes in metadata {}".format(
        #     self.att_weights.shape[0], self.num_attributes
        # )

        # Fix some evaluation consideration for attributes
        self.cls_depend = cfg.EVALUATION_ATTRIBUTE.CLS_DEPENDENT
        self.min_iou_match = cfg.EVALUATION_ATTRIBUTE.MIN_IOU_MATCH
        self.include_missing_boxes = cfg.EVALUATION_ATTRIBUTE.INCLUDE_MISSING_BOXES
        self.evaluate_contidional = cfg.EVALUATION_ATTRIBUTE.CONDITIONAL
        # self.embedding_contidional = cfg.EVALUATION_ATTRIBUTE.EMBEDDING_DICTIONARY

    def reset(self):
        self._labels = []
        self._predictions = []
        self._attributes = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                if self.evaluate_contidional:
                    include_fields = ["att_scores", "noun_scores", "cond_att_scores"]
                else:
                    include_fields = ["att_scores"]
                prediction["instances"] = instances_to_coco_json_plus_fields(
                    instances, input["image_id"], fields=include_fields
                )
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)

            if len(prediction) > 1:
                output_height, output_width = output["instances"].image_size
                prediction["height"] = output_height
                prediction["width"] = output_width

                self._predictions.append(prediction)

                # Fix size to the output size so that the overlap is correctly calculated
                input_height, input_width = input["instances"].image_size
                output_height, output_width = output["instances"].image_size
                gt_boxes = input["instances"].gt_boxes.tensor.clone()
                gt_boxes[:, 0] = gt_boxes[:, 0] / input_width * output_width
                gt_boxes[:, 2] = gt_boxes[:, 2] / input_width * output_width
                gt_boxes[:, 1] = gt_boxes[:, 1] / input_height * output_height
                gt_boxes[:, 3] = gt_boxes[:, 3] / input_height * output_height

                # Save label
                label = {
                    "image_id": input["image_id"],
                    "num_instances": len(input["instances"]),
                    "id_instances": input["instances"].id.tolist(),
                    "gt_boxes": gt_boxes.tolist(),
                    "gt_classes": input["instances"].gt_classes.tolist(),
                    "gt_att_vec": input["instances"].att_vec.to(torch.int).tolist(),
                }
                self._labels.append(label)

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        attributes = []
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            attributes = comm.gather(self._labels, dst=0)
            attributes = list(itertools.chain(*attributes))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
            attributes = self._labels

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(predictions, attributes, img_ids=img_ids)

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _tasks_from_predictions(self, predictions):
        """
        Get COCO API "tasks" (i.e. iou_type) from COCO-format predictions.
        """
        tasks = {"bbox"}
        for pred in predictions:
            if "segmentation" in pred:
                tasks.add("segm")
            if "keypoints" in pred:
                tasks.add("keypoints")
            if "att_scores" in pred:
                tasks.add("attributes")
        return sorted(tasks)

    def _eval_predictions(self, predictions, attributes=None, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(
            itertools.chain(*[x["instances"] for x in copy.deepcopy(predictions)])
        )
        for pred_instance in coco_results:
            _ = pred_instance.pop("features", None)

        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = (
                self._metadata.thing_dataset_id_to_contiguous_id
            )
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert (
                min(all_contiguous_ids) == 0
                and max(all_contiguous_ids) == num_classes - 1
            )

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            if task == "attributes":
                assert attributes is not None, f"Got task {task} but no attributes_dict"
                # Unconditional evaluation
                self._logger.info("Doing Unconditional Prompt Evaluation")
                att_eval = evaluate_attribute_detection(
                    predictions,
                    attributes,
                    self._metadata,
                    self._output_dir,
                    ["mAP"],
                    return_gt_pred_pair=False,
                    cls_depend=self.cls_depend,
                    min_iou_match=self.min_iou_match,
                    include_missing_boxes=self.include_missing_boxes,
                    conditional=False,
                )
                res = self._derive_att_results(att_eval)
                if len(res) == 0:
                    self._logger.info("No instances predicted for all dataset.")

                if self.evaluate_contidional:
                    # Conditional evaluation
                    self._logger.info("Doing Conditional Prompt Evaluation")
                    att_eval_cond = evaluate_attribute_detection(
                        predictions,
                        attributes,
                        self._metadata,
                        self._output_dir,
                        ["mAP"],
                        return_gt_pred_pair=False,
                        cls_depend=self.cls_depend,
                        min_iou_match=self.min_iou_match,
                        include_missing_boxes=self.include_missing_boxes,
                        conditional=True,
                    )
                    uncond_res = {key + "_uncond": val for key, val in res.items()}
                    cond_res = self._derive_att_results(att_eval_cond)
                    cond_res = {key + "_cond": val for key, val in cond_res.items()}
                    res = uncond_res
                    res.update(cond_res)
                    # res = self._derive_att_results(att_eval_cond)
                    if len(cond_res) == 0:
                        self._logger.info("No instances predicted for all dataset.")

            else:
                assert task in {
                    "bbox",
                    "segm",
                    "keypoints",
                }, f"Got unknown task: {task}!"
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api,
                        coco_results,
                        task,
                        kpt_oks_sigmas=self._kpt_oks_sigmas,
                        use_fast_impl=self._use_fast_impl,
                        img_ids=img_ids,
                    )
                    if len(coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )

                res = self._derive_coco_results(
                    coco_eval, task, class_names=self._metadata.get("thing_classes")
                )
            self._results[task] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Additionally
        - plot mAP50 for 'base classes' and 'novel classes'
        - plot mAP50 per class
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(
                coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan"
            )
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type)
            + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        base_names = set([x["name"] for x in categories_base])
        novel_names = set([x for x in class_names if x not in base_names])
        results_per_category = []
        results_per_category50 = []
        results_per_category50_base = []
        results_per_category50_novel = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.nanmean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))
            precision50 = precisions[0, :, idx, 0, -1]
            precision50 = precision50[precision50 > -1]
            ap50 = np.nanmean(precision50) if precision50.size else float("nan")
            results_per_category50.append(("{}".format(name), float(ap50 * 100)))
            if name in base_names:
                results_per_category50_base.append(float(ap50 * 100))
            if name in novel_names:
                results_per_category50_novel.append(float(ap50 * 100))

        self._logger.info(
            "Evaluation results for AP50 \n"
            + create_small_table(
                {
                    "results_base": np.nanmean(results_per_category50_base),
                    "results_novel": np.nanmean(results_per_category50_novel),
                    "results_generalized": np.nanmean(
                        results_per_category50_base + results_per_category50_novel
                    ),
                }
            )
        )

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)]
        )
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        N_COLS = min(6, len(results_per_category50) * 2)
        results_flatten = list(itertools.chain(*results_per_category50))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)]
        )
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP50"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP50: \n".format(iou_type) + table)
        # self._logger.info(
        #     "base {} AP50: {}".format(
        #         iou_type,
        #         np.nanmean(results_per_category50_base),
        #     )
        # )
        # self._logger.info(
        #     "novel {} AP50: {}".format(
        #         iou_type,
        #         np.nanmean(results_per_category50_novel),
        #     )
        # )

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        results.update({"AP50-" + name: ap for name, ap in results_per_category50})
        results["AP50_base"] = sum(results_per_category50_base) / len(
            results_per_category50_base
        )
        results["AP50_novel"] = sum(results_per_category50_novel) / len(
            results_per_category50_novel
        )
        return results

    def _derive_att_results(self, att_eval):
        # all_metrics = ["OVt{thr}_f1", "OVt{thr}_P", "OVt{thr}_R", "OVt{thr}_tnr", "PCt{thr}_ap", "PCt{thr}_f1", \
        #     "PCt{thr}_P", "PCt{thr}_R", "PCt{thr}_bacc", "OV@{top_k}_f1", "OV@{top_k}_P", "OV@{top_k}_R", \
        #     "OV@{top_k}_tnr", "PC@{top_k}_ap", "PC@{top_k}_f1", "PC@{top_k}_P", "PC@{top_k}_R", "PC@{top_k}_bacc"]
        all_metrics = [
            "PCrand_P",
            "PCt{thr}_ap",
            # "PCt{thr}_R",
            # "PCt{thr}_P",
            # "PCt{thr}_f1",
            "PCt{thr}_bacc",
            # "PC@{top_k}_R",
            # "PC@{top_k}_P",
            # "PC@{top_k}_f1",
            # "OVt{thr}_R",
            # "OVt{thr}_P",
            # "OVt{thr}_f1",
            # "OVt{thr}_tnr",
            # "OV@{top_k}_R",
            # "OV@{top_k}_P",
            # "OV@{top_k}_f1",
            # "OV@{top_k}_tnr",
        ]

        rnd_results = {}
        att_type_idx = {}
        table_results = {
            "Type": [],
        }
        metrics = []
        for m in all_metrics:
            m_name = m.format(thr=att_eval.threshold, top_k=att_eval.top_k)
            metrics.append(m_name)
            table_results[m_name] = []

        results = att_eval.print_evaluation()

        return_metrics = {}
        for metric_des, score in results.items():
            if len(metric_des.split("/")) == 3:
                metric_type, att_type, metric = metric_des.split("/")
            elif len(metric_des.split("/")) == 4:
                metric_type, _, att_type, metric = metric_des.split("/")
            metric_type = metric_type.replace("_", "")
            metric = metric.replace("precision", "P").replace("recall", "R")
            metric_name = metric_type + "_" + metric
            if "rand" in metric_name and "PCt" in metric_name:
                metric_name = "PCrand_" + metric_name.split("_")[-1]

            if metric_name not in table_results.keys():
                continue

            if att_type not in table_results["Type"]:
                att_type_idx[att_type] = len(table_results["Type"])
                table_results["Type"].append(att_type)

            assert len(table_results[metric_name]) == att_type_idx[att_type]

            table_results[metric_name].append(score)

            if "ap" in metric_name:
                return_metrics[metric_name + "/" + att_type] = score

        # # add random scores
        # for att_type, metric_scores in rnd_results.items():
        #     table_results["Type"] = [att_type]+table_results["Type"]
        #     for metric_name, score in table_results.items():
        #         if metric_name in metric_scores.keys():
        #             table_results[metric_name] = [metric_scores[metric_name]]+table_results[metric_name]
        #         elif metric_name!="Type":
        #             table_results[metric_name] = ['']+table_results[metric_name]

        table = tabulate(
            table_results,
            headers="keys",
            tablefmt="pipe",
            floatfmt=".3f",
            numalign="left",
        )
        self._logger.info("Per-att_group: \n" + table)

        return return_metrics

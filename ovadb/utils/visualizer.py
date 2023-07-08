import numpy as np
import matplotlib.colors as mplc
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
)
from detectron2.utils.visualizer import ColorMode, _create_text_labels
from detectron2.utils.visualizer import Visualizer as _Visualizer
from detectron2.utils.colormap import random_color
from detectron2.utils.visualizer import VisImage


def _create_text_labels_extended(
    classes,
    scores,
    class_names,
    is_crowd=None,
    attributes=None,
    scores_att=None,
    att_threshole=0.5,
    only_pos=True,
):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):
        attributes (list[str] or None):
        scores_att (list[float] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    if attributes is not None and scores_att is not None:
        attributes = ["/".join(att.split("/")[:2]) for att in attributes]

        if att_threshole > 1:
            for anidx, s in enumerate(scores_att):
                # pick top k pos and neg
                sort_indx = s.argsort()
                sort_indx = sort_indx[::-1]
                pos_idx = sort_indx[:att_threshole]

                pos_att = []
                s_att = []
                pos_text = ""
                for idx in pos_idx:
                    if s[idx] > 0:
                        pos_att.append(attributes[idx])
                        s_att.append(s[idx])
                        # pos_text += attributes[idx] + "{:.0f}".format(s[idx] * 100) + ",\n   "
                        pos_text += attributes[idx] + ",\n   "

                labels[anidx] += " " + str(anidx) + "\n|+:{}".format(pos_text)

                # neg_idx = sort_indx[att_threshole:]
                # import ipdb; ipdb.set_trace()
                # pos_att = []
                # neg_att = []
                # s_att = []
                # labels[anidx] += (
                #     " "
                #     + str(anidx)
                #     + "\n|+:{}".format(
                #         ",\n   ".join(
                #             [
                #                 attributes[idx] + ("{:.0f}".format(s[idx] * 100) if s[idx]>0 else "")
                #                 for idx in pos_idx
                #             ]
                #         )
                #     )
                # )
                # if not only_pos:
                #     labels[anidx] += "\n|-:{}".format(
                #         ",\n   ".join(
                #             [
                #                 attributes[idx] + "{:.0f}".format(s[idx] * 100)
                #                 for idx in sort_indx[-att_threshole:]
                #             ]
                #         )
                #     )
        else:
            for anidx, s in enumerate(scores_att):
                pos_att = []
                neg_att = []
                s_att = []
                for idx in range(len(s)):
                    if s[idx] > att_threshole:
                        pos_att.append(attributes[idx])
                        s_att.append(s[idx])
                    if s[idx] <= att_threshole and s[idx] >= 0:
                        neg_att.append(attributes[idx])
                        s_att.append(s[idx])
                labels[anidx] += (
                    " " + str(anidx) + "\n|+:{}".format(",\n   ".join(pos_att))
                    if len(pos_att) > 0
                    else ""
                )
                if not only_pos:
                    labels[anidx] += (
                        "\n|-:{}".format(",\n   ".join(neg_att))
                        if 0 < len(pos_att) < 10
                        else ""
                    )

        labels_class = [label.split("|")[0].strip() for label in labels]
        return labels_class, labels

    return labels


def _create_text_labels_extended_ann_and_pred(
    classes,
    pred_classes,
    scores,
    class_names,
    attributes,
    scores_att,
    attribute_names,
):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):
        attributes (list[int] or None):
        scores_att (list[float] or None):

    Returns:
        list[str] or None
    """
    assert len(classes) == len(scores) and len(scores) == len(pred_classes)
    labels = [
        class_names[i] + "/{}-{:.0f}%".format(class_names[j], ps * 100)
        for i, ps, j in zip(classes, scores, pred_classes)
    ]

    assert (
        len(attributes) == len(scores_att)
        and len(attribute_names) == len(scores_att[0])
        and len(attribute_names) == len(attributes[0])
    )
    attributes_labels = []
    for idx, (obj_att, att_scores) in enumerate(zip(attributes, scores_att)):
        sample_att_labels = ""
        for att, active, score in zip(attribute_names, obj_att, att_scores):
            if active == 1:
                att_str = "+ "
            elif active == 0:
                att_str = "- "
            else:
                att_str = "? "
            att_str += "({:.0f}%) ".format(score * 100)
            att_str += "/".join(att.split("/")[:1]) + ",\n   "
            sample_att_labels += att_str

        labels[idx] += " " + str(idx) + "\n|{}".format(sample_att_labels)

    labels_class = [label.split("|")[0].strip() for label in labels]
    return labels_class, labels


class Visualizer(_Visualizer):
    def draw_proposals_dataset_dict(self, dic, num_prop=20):
        """
        Draw extended annotations in Detectron2 Dataset format.

        Args:
            dic (dict): annotation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = dic["proposal_boxes"]
        scores = dic["proposal_objectness_logits"]
        box_mode = dic["proposal_bbox_mode"]
        boxes = BoxMode.convert(boxes, box_mode, BoxMode.XYXY_ABS)
        labels = ["{:.0f}%".format(s * 100) for s in scores]
        self.overlay_instances(labels=labels[:num_prop], boxes=boxes[:num_prop])

        return self.output

    def draw_caption_dataset_dict(self, dic, scale=0.9):
        """
        Draw extended annotations in Detectron2 Dataset format.

        Args:
            dic (dict): annotation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        # Get captions
        captions = dic.get("caption", None)
        text = "\n".join(captions)
        # Extend image to  write captions under it
        # height, width, chan = self.img.shape
        # cap_image = np.zeros((15*len(captions)+5, width, chan)).astype(np.uint8)
        cap_image = np.zeros((15,) + self.img.shape[1:]).astype(np.uint8)
        cap_output = VisImage(cap_image, scale=self.output.scale)
        cap_output.ax.text(
            0.0,
            5.0,
            text,
            size=self._default_font_size * self.output.scale * scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment="left",
            color=np.maximum(list(mplc.to_rgb("white")), 0.2),
            zorder=10,
            rotation=0,
        )
        cap_image = cap_output.get_image()
        original_image = self.output.get_image()
        image_text = np.concatenate([original_image, cap_image], axis=0)
        image_text = VisImage(image_text, scale=self.output.scale)

        return image_text

    def draw_att_text_black(self, att_txt_list, scale=0.8, colors=None):
        """
        Draw extended annotations in Detectron2 Dataset format.

        Args:
            dic (dict): annotation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        # Get captions
        n_att = [att_txt.count("\n") for att_txt in att_txt_list]
        # Extend image to  write text under it
        height, width, chan = self.img.shape

        line_height = self._default_font_size * 1.4
        columns_att = 3
        caption_height = int(
            line_height * max(n_att) * max(1, 1 + len(att_txt_list) // columns_att) + 5
        )

        cap_image = np.zeros((caption_height, width, chan)).astype(np.uint8)
        # cap_image = np.zeros((15,)+self.img.shape[1:]).astype(np.uint8)
        cap_output = VisImage(cap_image, scale=self.output.scale)
        for idx, text in enumerate(att_txt_list):
            if colors:
                color = colors[idx]
            else:
                color = np.maximum(list(mplc.to_rgb("white")), 0.2)

            x_coord = width / columns_att * (idx % columns_att)
            y_coord = 5.0 + line_height * max(n_att) * int(idx // columns_att)
            # print("x", x_coord)
            # print("y", y_coord)
            # print("text", text)
            cap_output.ax.text(
                x_coord,
                y_coord,
                text,
                size=self._default_font_size * self.output.scale * scale,
                family="sans-serif",
                bbox={
                    "facecolor": "black",
                    "alpha": 0.8,
                    "pad": 0.7,
                    "edgecolor": "none",
                },
                verticalalignment="top",
                horizontalalignment="left",
                color=color,
                zorder=10,
                rotation=0,
            )
        cap_image = cap_output.get_image()
        original_image = self.output.get_image()
        image_text = np.concatenate([original_image, cap_image], axis=0)
        image_text = VisImage(image_text, scale=self.output.scale)

        return image_text

    def draw_att_text(self, att_txt_list, scale=0.8, colors=None):
        """
        Draw extended annotations in Detectron2 Dataset format.

        Args:
            dic (dict): annotation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        # Get captions
        n_att = [att_txt.count("\n") for att_txt in att_txt_list]
        # Extend image to  write text under it
        height, width, chan = self.img.shape

        line_height = self._default_font_size * 1.4
        columns_att = 3
        caption_height = int(
            line_height * max(n_att) * max(1, 1 + len(att_txt_list) // columns_att) + 5
        )

        cap_image = (np.ones((caption_height, width, chan)) * 255).astype(np.uint8)
        # cap_image = np.zeros((15,)+self.img.shape[1:]).astype(np.uint8)
        cap_output = VisImage(cap_image, scale=self.output.scale)
        for idx, text in enumerate(att_txt_list):
            # if colors:
            #     color = colors[idx]
            # else:
            #     color = np.maximum(list(mplc.to_rgb("white")), 0.2)
            color = list(mplc.to_rgb("black"))

            x_coord = width / columns_att * (idx % columns_att)
            y_coord = 5.0 + line_height * max(n_att) * int(idx // columns_att)
            # print("x", x_coord)
            # print("y", y_coord)
            # print("text", text)
            cap_output.ax.text(
                x_coord,
                y_coord,
                text,
                size=self._default_font_size * self.output.scale * scale,
                family="sans-serif",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.8,
                    "pad": 0.7,
                    "edgecolor": "none",
                },
                verticalalignment="top",
                horizontalalignment="left",
                color=color,
                zorder=10,
                rotation=0,
            )
        cap_image = cap_output.get_image()
        original_image = self.output.get_image()
        image_text = np.concatenate([original_image, cap_image], axis=0)
        image_text = VisImage(image_text, scale=self.output.scale)

        return image_text

    def old_draw_att_text(self, att_txt_list, scale=0.8, colors=None):
        """
        Draw extended annotations in Detectron2 Dataset format.

        Args:
            dic (dict): annotation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        # Get captions
        n_att = [att_txt.count("\n") for att_txt in att_txt_list]
        # Extend image to  write text under it
        height, width, chan = self.img.shape
        cap_image = np.zeros((25 * max(n_att) + 5, width, chan)).astype(np.uint8)
        # cap_image = np.zeros((15,)+self.img.shape[1:]).astype(np.uint8)
        cap_output = VisImage(cap_image, scale=self.output.scale)
        for idx, text in enumerate(att_txt_list):
            if colors:
                color = colors[idx]
            else:
                color = np.maximum(list(mplc.to_rgb("white")), 0.2)

            cap_output.ax.text(
                0.0 + width / len(att_txt_list) * idx,
                5.0,
                text,
                size=self._default_font_size * self.output.scale * scale,
                family="sans-serif",
                bbox={
                    "facecolor": "black",
                    "alpha": 0.8,
                    "pad": 0.7,
                    "edgecolor": "none",
                },
                verticalalignment="top",
                horizontalalignment="left",
                color=color,
                zorder=10,
                rotation=0,
            )
        cap_image = cap_output.get_image()
        original_image = self.output.get_image()
        image_text = np.concatenate([original_image, cap_image], axis=0)
        image_text = VisImage(image_text, scale=self.output.scale)

        return image_text

    def draw_dataset_dict(self, dic, font_size=1.0):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.

        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        annos = dic.get("annotations", None)
        _default_font_size = self._default_font_size
        self._default_font_size = _default_font_size * font_size
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [
                BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                if len(x["bbox"]) == 4
                else x["bbox"]
                for x in annos
            ]

            colors = None
            category_ids = [x["category_id"] for x in annos]
            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get(
                "thing_colors"
            ):
                colors = [
                    self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
                    for c in category_ids
                ]
            else:
                colors = [random_color(rgb=True, maximum=1) for _ in range(len(annos))]
            colors = [np.clip(color + 0.15, 0.25, 1.0) for color in colors]
            names = self.metadata.get("thing_classes", None)

            if "att_vec" not in annos[0].keys() and hasattr(
                self.metadata, "attributes_dict"
            ):
                attribute_classes = [
                    att["name"] for att in self.metadata.attributes_dict["attributes"]
                ]
                self.metadata.get("attribute_classes", None)

                ann_vecs = []
                for ann in annos:
                    if hasattr(self.metadata.attributes_dict, "ann_id_to_patch_id"):
                        if (
                            ann["id"]
                            in self.metadata.attributes_dict[
                                "ann_id_to_patch_id"
                            ].keys()
                        ):
                            ann_id = self.metadata.attributes_dict[
                                "ann_id_to_patch_id"
                            ][ann["id"]]
                        else:
                            ann_id = -1
                        ann_id = self.metadata.attributes_dict["ann_id_to_patch_id"][
                            ann["id"]
                        ]
                    if ann_id in self.metadata.attributes_dict["ann_vecs"].keys():
                        ann_vec = self.metadata.attributes_dict["ann_vecs"][ann_id]
                    else:
                        ann_vec = np.zeros(len(attribute_classes))
            attribute_classes = self.metadata.get("attribute_classes", None)

            labels = _create_text_labels_extended(
                category_ids,
                scores=None,
                class_names=names,
                is_crowd=[x.get("iscrowd", 0) for x in annos],
                attributes=attribute_classes,
                scores_att=[x.get("att_vec", None) for x in annos],
            )
            if isinstance(labels, tuple) and len(labels) == 2:
                labels, labels_att = labels
            else:
                labels_att = None

            self.overlay_instances(
                labels=labels,
                boxes=boxes,
                masks=masks,
                keypoints=keypts,
                assigned_colors=colors,
            )
            if labels_att:
                return self.draw_att_text(labels_att, colors=colors)
        self._default_font_size = _default_font_size

        return self.output

    def draw_dataset_dict_predictions(self, dic, font_size=1.0, att_threshole=8):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.

        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        # annos = dic.get("annotations", None)
        annos = dic.get("predictions", None)
        _default_font_size = self._default_font_size
        self._default_font_size = _default_font_size * font_size
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [
                BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                if len(x["bbox"]) == 4
                else x["bbox"]
                for x in annos
            ]

            colors = None
            category_ids = [x["category_id"] for x in annos]
            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get(
                "thing_colors"
            ):
                colors = [
                    self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
                    for c in category_ids
                ]
            else:
                colors = [random_color(rgb=True, maximum=1) for _ in range(len(annos))]
            colors = [np.clip(color + 0.15, 0.25, 1.0) for color in colors]
            names = self.metadata.get("thing_classes", None)

            attribute_classes = self.metadata.get("attribute_classes", None)
            labels = _create_text_labels_extended(
                category_ids,
                scores=[x.get("score", None) for x in annos],
                class_names=names,
                is_crowd=[x.get("iscrowd", 0) for x in annos],
                attributes=attribute_classes,
                scores_att=[x.get("att_scores", None) for x in annos],
                att_threshole=att_threshole,
            )
            if isinstance(labels, tuple) and len(labels) == 2:
                labels, labels_att = labels
            else:
                labels_att = None

            self.overlay_instances(
                labels=labels,
                boxes=boxes,
                masks=masks,
                keypoints=keypts,
                assigned_colors=colors,
            )
            if labels_att:
                return self.draw_att_text(labels_att, colors=colors)
        self._default_font_size = _default_font_size

        return self.output

    def draw_attribute_predictions(self, predictions, font_size=1.0):
        _default_font_size = self._default_font_size
        self._default_font_size = _default_font_size * font_size

        boxes = predictions.get("box_pred", None)
        boxes = [b.tensor[0].numpy() for b in boxes]
        scores = predictions.get("score", None)
        category_ids = predictions.get("category_id", None)
        thing_classes = self.metadata.get("thing_classes", None)
        attribute_classes = self.metadata.get("attribute_classes", None)
        scores_att = predictions.get("att_scores", None)

        labels = _create_text_labels_extended(
            category_ids,
            scores=scores,
            class_names=thing_classes,
            is_crowd=None,
            attributes=attribute_classes,
            scores_att=scores_att,
            att_threshole=8,
        )

        colors = None
        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get(
            "thing_colors"
        ):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
                for c in category_ids
            ]

        self.overlay_instances(labels=labels, boxes=boxes, assigned_colors=colors)
        self._default_font_size = _default_font_size

        return self.output

    def draw_instances(self, instances):
        boxes = instances.gt_boxes if instances.has("gt_boxes") else None
        classes = instances.gt_classes.tolist() if instances.has("gt_classes") else None
        labels = _create_text_labels(
            classes, None, self.metadata.get("thing_classes", None)
        )

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get(
            "thing_colors"
        ):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
                for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        self.overlay_instances(
            boxes=boxes,
            labels=labels,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output

    def draw_attribute_instances(self, instances, font_size=1.0):
        _default_font_size = self._default_font_size
        self._default_font_size = _default_font_size * font_size

        boxes = instances.gt_boxes if instances.has("gt_boxes") else None
        att_vec = instances.att_vec if instances.has("att_vec") else None
        classes = instances.gt_classes.tolist() if instances.has("gt_classes") else None
        labels = _create_text_labels_extended(
            classes,
            scores=None,
            class_names=self.metadata.get("thing_classes", None),
            is_crowd=None,
            attributes=self.metadata.get("attribute_classes", None),
            scores_att=att_vec,
            att_threshole=0.9,
        )

        colors = None
        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get(
            "thing_colors"
        ):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
                for c in category_ids
            ]

        self.overlay_instances(labels=labels, boxes=boxes, assigned_colors=colors)
        self._default_font_size = _default_font_size

        return self.output

    def draw_image_dict_instance(self, dic, fpath, font_size=1.0, att_threshole=8):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.

        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        annos = dic.get("annotations", None)
        preds = dic.get("predictions", None)

        _default_font_size = self._default_font_size
        self._default_font_size = _default_font_size * font_size
        # annos = [annotations[idx]]
        # preds = [predictions[idx]]

        boxes_anns = [
            BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
            if len(x["bbox"]) == 4
            else x["bbox"]
            for x in annos
        ]
        boxes_preds = [
            BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
            if len(x["bbox"]) == 4
            else x["bbox"]
            for x in preds
        ]
        colors_anns = [np.array([0.0, 1.0, 0.0])] * len(annos)
        colors_pred = [np.array([0.0, 0.0, 1.0])] * len(preds)
        class_names = self.metadata.get("thing_classes", None)
        attribute_classes = self.metadata.get("attribute_classes", None)

        # import ipdb

        # ipdb.set_trace()
        labels = _create_text_labels_extended_ann_and_pred(
            [x["category_id"] for x in annos],
            [x["category_id"] for x in preds],
            [x.get("score", None) for x in preds],
            class_names,
            [x.get("att_vec", None) for x in annos],
            [x.get("att_scores", None) for x in preds],
            attribute_classes,
        )

        if isinstance(labels, tuple) and len(labels) == 2:
            labels, labels_att = labels
        else:
            labels_att = None

        self.overlay_instances(
            labels=labels,
            boxes=boxes_anns,
            masks=None,
            keypoints=None,
            assigned_colors=colors_anns,
        )
        self.overlay_instances(
            labels=None,
            boxes=boxes_preds,
            masks=None,
            keypoints=None,
            assigned_colors=colors_pred,
        )
        if labels_att:
            return self.draw_att_text(labels_att, colors=colors_anns)

        self._default_font_size = _default_font_size

        return self.output

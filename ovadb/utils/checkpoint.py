import logging
import os
import pickle
import torch
import json
import shutil
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple
from collections import OrderedDict

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.checkpoint.c2_model_loading import (
    convert_basic_c2_names,
    convert_c2_detectron_names,
    align_and_update_state_dicts,
)
from torch.nn.parallel import DistributedDataParallel


class CustomCheckpointer(DetectionCheckpointer):
    def resume_or_load_renaming_keys(
        self,
        path: str,
        *,
        resume: bool = True,
        rename_keys: Optional[Dict[str, Any]] = {},
    ) -> Dict[str, Any]:
        """
        If `resume` is True, this method attempts to resume from the last
        checkpoint, if exists. Otherwise, load checkpoint from the given path.
        This is useful when restarting an interrupted training job.
        Args:
            path (str): path to the checkpoint.
            resume (bool): if True, resume from the last checkpoint if it exists
                and load the model together with all the checkpointables. Otherwise
                only load the model without loading any checkpointables.
        Returns:
            same as :meth:`load`.
        """
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
            return self.load(path)
        else:
            if len(rename_keys) > 0:
                return self.load_renamed(
                    path, checkpointables=[], rename_keys=rename_keys
                )
            else:
                return self.load(path, checkpointables=[])

    def load_renamed(
        self,
        path,
        checkpointables: Optional[List[str]] = None,
        rename_keys: Optional[Dict[str, Any]] = {},
    ) -> Dict[str, Any]:
        need_sync = False

        if path and isinstance(self.model, DistributedDataParallel):
            logger = logging.getLogger(__name__)
            path = self.path_manager.get_local_path(path)
            has_file = os.path.isfile(path)
            all_has_file = comm.all_gather(has_file)
            if not all_has_file[0]:
                raise OSError(f"File {path} not found on main worker.")
            if not all(all_has_file):
                logger.warning(
                    f"Not all workers can read checkpoint {path}. "
                    "Training may fail to fully resume."
                )
                # TODO: broadcast the checkpoint file contents from main
                # worker, and load from it instead.
                need_sync = True
            if not has_file:
                path = None  # don't load if not readable

        if not path:
            # no checkpoint provided
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("[Checkpointer] Loading from {} ...".format(path))
        if not os.path.isfile(path):
            path = self.path_manager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        checkpoint = self._load_file(path)
        # If model comes from Imagenet then first convert names
        if "ImageNetPretrained" in path:
            renamed_check_point = align_and_update_state_dicts(
                self.model.state_dict(),
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            checkpoint = {"model": renamed_check_point}
            del renamed_check_point

        # If model changes but some keys should be conserved then these are specify in rename_keys
        if len(rename_keys) > 0:
            renamed_check_point = OrderedDict()
            # actual_model = self.model.state_dict()
            for k, v in checkpoint["model"].items():
                for old_k, new_k in rename_keys.items():
                    if old_k in k:
                        if isinstance(new_k, list):
                            for new_ki in new_k:
                                save_k = k.replace(old_k, new_ki)
                                renamed_check_point[save_k] = v
                        else:
                            save_k = k.replace(old_k, new_k)
                            renamed_check_point[save_k] = v
                    else:
                        renamed_check_point[k] = v
            checkpoint["model"] = renamed_check_point
            del renamed_check_point

        incompatible = self._load_model(checkpoint)
        if (
            incompatible is not None
        ):  # handle some existing subclasses that returns None
            self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:
                self.logger.info("Loading {} from {} ...".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))

        ret = checkpoint

        if need_sync:
            logger.info("Broadcasting model states from main worker ...")
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
        return ret

    def save_no_tag(self, name: str, **kwargs: Any) -> None:
        """
        Same as save but without modifying last_checkpoint file
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, f)

    def save_best_metric(self, name, test_scores, best_score, iteration=-1) -> Tuple:
        # find complete name of best metric
        metric_name, last_best_score = best_score

        # check if there is previous best model save
        model_name = os.path.join(self.save_dir, "{}.pth".format(name))
        json_name = os.path.join(self.save_dir, "{}.json".format(name))

        all_metrics_with_name = [
            m_name for m_name in test_scores.keys() if metric_name in m_name
        ]
        if len(all_metrics_with_name) == 0:
            return best_score
        if len(all_metrics_with_name) == 1:
            new_metric_name = all_metrics_with_name[0]
        elif len(all_metrics_with_name) > 1:
            unseen_set = [
                m_name
                for m_name in all_metrics_with_name
                if ("general_" not in m_name and "not_" not in m_name)
            ]
            if len(unseen_set) > 0:
                new_metric_name = unseen_set[0]
        else:
            new_metric_name = all_metrics_with_name[0]

        arguments = {}
        arguments["metric_name"] = new_metric_name
        arguments["best_score"] = test_scores[new_metric_name]
        arguments["iteration"] = iteration
        arguments["model_name"] = model_name

        # if metric names changes keep both models
        if (
            os.path.isfile(model_name)
            and os.path.isfile(json_name)
            and last_best_score == -1
        ):
            checkpoint = self._load_file(model_name)
            checkpoint_arg = json.load(open(json_name, "r"))

            if new_metric_name != checkpoint_arg["metric_name"]:
                shutil.copy(
                    model_name,
                    os.path.join(
                        os.path.dirname(model_name),
                        checkpoint_arg["metric_name"].replace("/", "_")
                        + "_"
                        + os.path.basename(model_name),
                    ),
                )
                shutil.copy(
                    json_name,
                    os.path.join(
                        os.path.dirname(json_name),
                        checkpoint_arg["metric_name"].replace("/", "_")
                        + "_"
                        + os.path.basename(json_name),
                    ),
                )
                last_best_score = -1
            else:
                last_best_score = max(last_best_score, checkpoint_arg["best_score"])

        new_best_score = test_scores[new_metric_name]

        if new_best_score > last_best_score and iteration > 0:
            self.save_no_tag(name)
            json.dump(arguments, open(json_name, "w"))

            return (new_metric_name, new_best_score)

        last_best_score = max(last_best_score, new_best_score)
        return (new_metric_name, last_best_score)

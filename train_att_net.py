import logging
import os
import sys
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime
import GPUtil
from ast import literal_eval

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
    LVISEvaluator,
    COCOEvaluator,
)
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import (
    build_detection_train_loader,
    _train_loader_from_config,
)
from detectron2.utils.logger import setup_logger
from torch.cuda.amp import GradScaler

from ovadb.data.custom_build_augmentation import build_custom_augmentation
from ovadb.data.custom_dataset_dataloader import (
    build_custom_train_loader,
    _custom_train_loader_from_config,
)
from ovadb.custom_solver import build_custom_optimizer
from ovadb.evaluation.custom_coco_eval import CustomCOCOEvaluator

from ovadb.data.custom_dataset_mapper import CustomDatasetMapper
from ovadb.config import add_ovadb_config
from ovadb.evaluation.attribute_evaluation import AttributeEvaluator
from ovadb.evaluation.ann_box_evaluator import inference_on_dataset_annotation_box
from ovadb.modeling.utils import (
    reset_cls_test,
    reset_attribute_test_model,
    restore_attribute_test_model,
)
from ovadb.utils.events import CustomMetricPrinter, CalcWriter
from ovadb.utils.checkpoint import CustomCheckpointer

logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    results = OrderedDict()
    for d, dataset_name in enumerate(cfg.DATASETS.TEST):
        mapper = CustomDatasetMapper(cfg, False)
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name)
        )
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if cfg.MODEL.RESET_CLS_TESTS:
            reset_cls_test(
                model,
                cfg.MODEL.TEST_CLASSIFIERS[d],
                cfg.MODEL.TEST_NUM_CLASSES[d],
                dataset_name,
            )

        if evaluator_type == "lvis":
            evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == "coco":
            if dataset_name in {"coco_val2017_ovd17_g", "coco_val2017_ovd32_g"}:
                # Additionally plot mAP for 'seen classes' and 'unseen classes'
                evaluator = CustomCOCOEvaluator(dataset_name, cfg, True, output_folder)
            else:
                evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif "attribute" in evaluator_type:
            reset_attribute_test_model(
                model,
                dataset_name,
                cfg.MODEL.TEST_CLASSIFIERS[d],
                cfg.MODEL.TEST_ATTRIBUTE_CLASSIFIERS[d],
            )
            evaluator = AttributeEvaluator(
                dataset_name,
                cfg,
                None,
                True,
                output_folder,
                noun_weights_path=cfg.MODEL.TEST_CLASSIFIERS[d],
                att_weights_path=cfg.MODEL.TEST_ATTRIBUTE_CLASSIFIERS[d],
            )

            if "boxann" in evaluator_type:
                # evaluate on labeled box
                results[dataset_name] = inference_on_dataset_annotation_box(
                    model, data_loader, evaluator
                )
            else:
                results[dataset_name] = inference_on_dataset(
                    model, data_loader, evaluator
                )
            restore_attribute_test_model(model)
        else:
            assert 0, evaluator_type

        if "attribute" not in evaluator_type:
            results[dataset_name] = inference_on_dataset(model, data_loader, evaluator)

        if comm.is_main_process():
            assert isinstance(
                results[dataset_name], dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results[dataset_name]
            )
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results[dataset_name])

    if len(results) == 1:
        results = list(results.values())[0]

    # Restore classes for training
    reset_cls_test(
        model,
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
        cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        cfg.DATASETS.TRAIN[0],
    )
    return results


def do_train(cfg, model, resume=False):
    last_eval_results = None
    reset_cls_test(
        model,
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
        cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        cfg.DATASETS.TRAIN[0],
    )
    model.train()
    if cfg.SOLVER.USE_CUSTOM_SOLVER:
        optimizer = build_custom_optimizer(cfg, model)
    else:
        assert cfg.SOLVER.OPTIMIZER == "SGD"
        assert cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != "full_model"
        assert cfg.SOLVER.BACKBONE_MULTIPLIER == 1.0
        optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = CustomCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    if cfg.MODEL.RENAME_KEYS_WEIGHTS:
        rename_keys = {
            "roi_heads.box_predictor.emb_pred": [
                "roi_heads.box_predictor.cls_score.linear"
            ],
        }
    else:
        rename_keys = {}
    start_iter = (
        checkpointer.resume_or_load_renaming_keys(
            cfg.MODEL.WEIGHTS, resume=resume, rename_keys=rename_keys
        ).get("iteration", -1)
        + 1
    )
    if not resume:
        start_iter = 0
    iteration = start_iter

    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        cfg.SOLVER.CHECKPOINT_PERIOD,
        max_iter=max_iter,
        max_to_keep=2,
    )

    writers = (
        [
            CustomMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    mapper = (
        CustomDatasetMapper(cfg, True)
        if cfg.WITH_ANNOTATION_TYPE or cfg.WITH_IMAGE_LABELS
        else DatasetMapper(cfg, True)
    )

    if cfg.DATALOADER.SAMPLER_TRAIN in [
        "TrainingSampler",
        "RepeatFactorTrainingSampler",
    ]:
        train_data_dict = _train_loader_from_config(cfg, mapper=mapper)
        data_loader = build_detection_train_loader(**train_data_dict)
        dataset_len = len(train_data_dict["dataset"])
    else:
        # supports different samplers for different type of data
        train_data_dict = _custom_train_loader_from_config(cfg, mapper=mapper)
        data_loader = build_custom_train_loader(**train_data_dict)
        dataset_len = len(train_data_dict["dataset"])

    epoch_iters = dataset_len // cfg.SOLVER.IMS_PER_BATCH
    calc_writer = (
        [CalcWriter(os.path.join(cfg.OUTPUT_DIR, "metrics_log.csv"), epoch_iters)]
        if comm.is_main_process()
        else []
    )
    best_metric = (cfg.TEST.SAVE_MODEL_BEST_METRIC, -1)

    if cfg.FP16:
        scaler = GradScaler()

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()

        # Test before starting training
        if cfg.TEST.EVAL_INIT and start_iter == 0:
            logger.info("Evaluating initial performance")
            last_eval_results = do_test(cfg, model)
            comm.synchronize()
            if comm.is_main_process():
                flatten_results = flatten_results_dict(last_eval_results)
                storage.put_scalars(**flatten_results)
                for writer in calc_writer:
                    writer.write()

        # Training loop
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()
            loss_dict = model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), (loss_dict, data)

            loss_dict_reduced = {
                k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            if cfg.FP16:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()

            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False
            )

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()

            if iteration % 100 == 0 and 0 < iteration < 300 + start_iter:
                GPUtil.showUtilization(all=True)

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                last_eval_results = do_test(cfg, model)
                comm.synchronize()
                if comm.is_main_process():
                    flatten_results = flatten_results_dict(last_eval_results)
                    storage.put_scalars(**flatten_results)
                    for writer in calc_writer:
                        writer.write()
                    best_metric = checkpointer.save_best_metric(
                        "model_best", flatten_results, best_metric, iteration
                    )

            # all writers
            if iteration - start_iter > 5 and (
                iteration % 20 == 0 or iteration == max_iter
            ):
                for writer in writers:
                    writer.write()
            # excell writer
            if iteration - start_iter > 5 and (
                iteration % (max_iter // 100) == 0 or iteration == max_iter
            ):
                for writer in calc_writer:
                    writer.write()

            if comm.is_main_process():
                periodic_checkpointer.step(iteration)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))
            )
        )
        # test at end of training
        if cfg.TEST.EVAL_PERIOD > 0:
            last_eval_results = do_test(cfg, model)
            comm.synchronize()
            if comm.is_main_process():
                flatten_results = flatten_results_dict(last_eval_results)
                storage.put_scalars(**flatten_results)
                for writer in calc_writer:
                    writer.write()
                best_metric = checkpointer.save_best_metric(
                    "model_best", flatten_results, best_metric, iteration
                )

            return last_eval_results
    return last_eval_results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ovadb_config(cfg)
    cfg.merge_from_file(args.config_file)
    literal_ops = []
    for x in args.opts:
        try:
            literal_ops.append(literal_eval(x))
        except (SyntaxError, ValueError):
            literal_ops.append(x)
    cfg.merge_from_list(literal_ops)
    # cfg.merge_from_list(args.opts)
    if "/auto" in cfg.OUTPUT_DIR or "_auto_" in cfg.OUTPUT_DIR:
        n_gpu = comm.get_world_size()
        file_name = os.path.basename(args.config_file)[:-5] + "_{}gpu".format(n_gpu)
        if cfg.OUTPUT_DIR.endswith("/auto"):
            cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace("/auto", "/{}".format(file_name))
        if cfg.OUTPUT_DIR.endswith("_auto_"):
            cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace("_auto_", "_{}".format(file_name))
        logger.info("OUTPUT_DIR: {}".format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="ovadb")
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        if cfg.MODEL.RENAME_KEYS_WEIGHTS:
            rename_keys = {
                "roi_heads.box_predictor.emb_pred": [
                    "roi_heads.box_predictor.cls_score.linear"
                ],
            }
        else:
            rename_keys = {}
        CustomCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load_renaming_keys(
            cfg.MODEL.WEIGHTS, resume=args.resume, rename_keys=rename_keys
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[comm.get_local_rank()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.FIND_UNUSED_PARAM,
        )

    return do_train(cfg, model, resume=args.resume)


if __name__ == "__main__":
    args = default_argument_parser()
    args = args.parse_args()
    if args.num_machines == 1:
        args.dist_url = "tcp://127.0.0.1:{}".format(
            torch.randint(11111, 60000, (1,))[0].item()
        )
    else:
        if args.dist_url == "host":
            args.dist_url = "tcp://{}:12345".format(os.environ["SLURM_JOB_NODELIST"])
        elif not args.dist_url.startswith("tcp"):
            tmp = os.popen(
                "echo $(scontrol show job {} | grep BatchHost)".format(args.dist_url)
            ).read()
            tmp = tmp[tmp.find("=") + 1 : -1]
            args.dist_url = "tcp://{}:12345".format(tmp)
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

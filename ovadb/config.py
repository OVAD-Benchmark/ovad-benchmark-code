# Edited from Detic
from detectron2.config import CfgNode as CN


def add_ovadb_config(cfg):
    _C = cfg

    # Language model
    _C.MODEL.TEXT_MODEL = "clip"  # It can be clip or bert or roberta
    _C.MODEL.TEXT_MODEL_OUTPUT_FEATURES = (
        "pooler_output"  # If text model is bert or roberta then select possible outputs
    )
    # ['pooler_output', 'cls_token', 'mean_encodings', 'mean_embeddings']

    # Attribute classifier
    _C.MODEL.ATTRIBUTE_ON = False
    _C.MODEL.ROI_BOX_HEAD.ATTRIBUTE_ACTIVE = False  # Turn on when using attributes
    _C.MODEL.ROI_BOX_HEAD.ATTRIBUTE_CONDITIONAL = False  # Decide wheather to use conditional or unconditional on noun attribute evaluation
    _C.MODEL.ROI_BOX_HEAD.ATTRIBUTE_WEIGHT_PATH = ""
    _C.MODEL.TEST_ATTRIBUTE_CLASSIFIERS = []
    _C.MODEL.ROI_HEADS.NUM_ATTRIBUTES = 117
    _C.MODEL.ROI_BOX_HEAD.NORM_TEMP_ATTRIBUTE = 50.0
    _C.MODEL.ROI_BOX_HEAD.ADD_BOX_FEATURES_PREDICTION = False
    # Attribute evaluation
    _C.EVALUATION_ATTRIBUTE = CN()
    _C.EVALUATION_ATTRIBUTE.CLS_DEPENDENT = False
    _C.EVALUATION_ATTRIBUTE.MIN_IOU_MATCH = 0.5
    _C.EVALUATION_ATTRIBUTE.INCLUDE_MISSING_BOXES = True
    _C.EVALUATION_ATTRIBUTE.CONDITIONAL = False
    _C.EVALUATION_ATTRIBUTE.EMBEDDING_DICTIONARY = ""
    _C.EVALUATION_ATTRIBUTE.CONDITIONAL_USE_OBJ_LABEL = False
    _C.EVALUATION_ATTRIBUTE.CONDITIONAL_TEMPERATURE = 50.0

    # Test metric to save best model
    _C.TEST.SAVE_MODEL_BEST_METRIC = ""
    _C.TEST.EVAL_INIT = True  # Test model at the begining of training

    # Part (box-phn) matching loss
    _C.MODEL.ROI_BOX_HEAD.PARTMATCH_BOX_SELECTION = "max_area"
    _C.MODEL.ROI_BOX_HEAD.PARTMATCH_LOSS = "max_score_per_phn"
    _C.MODEL.ROI_BOX_HEAD.PARTMATCH_TOPK_BOX = 3
    _C.MODEL.ROI_BOX_HEAD.PARTMATCH_TOPK_PHN = 10
    _C.MODEL.ROI_BOX_HEAD.PARTMATCH_WEIGHT = 1.0

    # For running object-centric-ovd
    _C.MODEL.ROI_BOX_HEAD.WEIGHT_TRANSFER = False
    _C.MODEL.ROI_BOX_HEAD.TWO_LAYER_BBOX_PRED = False
    
    # Add phrase nouns
    _C.MODEL.WITH_PHRASE_NOUNS = False
    _C.MODEL.SYNC_PHRASE_NOUNS_BATCH = False
    _C.MODEL.ROI_BOX_HEAD.PHRASE_NOUNS_WEIGHT = 1.0  # phrase loss weight
    _C.MODEL.ROI_BOX_HEAD.NEG_PHRASE_NOUNS_WEIGHT = 0.125  # phrase loss hyper-parameter
    _C.MODEL.LOAD_TEXT_FEATURES_FROM_MEMORY = False
    _C.MODEL.RENAME_KEYS_WEIGHTS = False
    _C.DATALOADER.CAPTION_FEATURES = ""
    _C.DATALOADER.PHRASE_FEATURES = ""
    _C.DATALOADER.NUM_NEG_CAPTIONS = 63  # gpus x bs x ncap = 8x8x1
    _C.DATALOADER.NUM_NEG_PHRASES = 50  # 640  # gpus x bs x nphn = 8x8x10

    # Add image labes
    _C.WITH_ANNOTATION_TYPE = False
    _C.WITH_IMAGE_LABELS = False  # Turn on co-training with classification data

    # Open-vocabulary classifier
    _C.MODEL.ROI_BOX_HEAD.PREDICTOR_NAME = "FastRCNNOutputLayers"
    _C.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS = (
        False  # Use fixed classifier for open-vocabulary detection
    )
    _C.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = (
        "datasets/metadata/lvis_v1_clip_a+cname.npy"
    )
    _C.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM = 512
    _C.MODEL.ROI_BOX_HEAD.NORM_WEIGHT = True
    _C.MODEL.ROI_BOX_HEAD.NORM_TEMP = 50.0
    _C.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS = False
    _C.MODEL.ROI_BOX_HEAD.USE_BIAS = 0.0  # >= 0: not use

    _C.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE = False  # CenterNet2
    _C.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    _C.MODEL.ROI_BOX_HEAD.PRIOR_PROB = 0.01
    _C.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False  # Federated Loss
    _C.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = (
        "datasets/metadata/lvis_v1_train_cat_info.json"
    )
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT = 50
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT = 0.5

    # Classification data configs
    _C.MODEL.ROI_BOX_HEAD.IMAGE_LABEL_LOSS = "max_size"  # max, softmax, sum
    _C.MODEL.ROI_BOX_HEAD.IMAGE_LOSS_WEIGHT = 0.1
    _C.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE = 1.0
    _C.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX = (
        False  # Used for image-box loss and caption loss
    )
    _C.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS = 128  # num proposals for image-labeled data
    _C.MODEL.ROI_BOX_HEAD.WITH_SOFTMAX_PROP = False  # Used for WSDDN
    _C.MODEL.ROI_BOX_HEAD.CAPTION_WEIGHT = 1.0  # Caption loss weight
    _C.MODEL.ROI_BOX_HEAD.NEG_CAP_WEIGHT = 0.125  # Caption loss hyper-parameter
    _C.MODEL.ROI_BOX_HEAD.ADD_FEATURE_TO_PROP = False  # Used for WSDDN
    _C.MODEL.ROI_BOX_HEAD.SOFTMAX_WEAK_LOSS = False  # Used when USE_SIGMOID_CE is False
    _C.MODEL.ROI_BOX_HEAD.BCE_AND_CE = False  # Used when using both binary cross entropy and cross entropy

    _C.MODEL.ROI_HEADS.MASK_WEIGHT = 1.0
    _C.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = False  # For demo only

    # Caption losses
    _C.MODEL.CAP_BATCH_RATIO = 4  # Ratio between detection data and caption data
    _C.MODEL.WITH_CAPTION = False
    _C.MODEL.SYNC_CAPTION_BATCH = (
        False  # synchronize across GPUs to enlarge # "classes"
    )

    # dynamic class sampling when training with 21K classes
    _C.MODEL.DYNAMIC_CLASSIFIER = False
    _C.MODEL.NUM_SAMPLE_CATS = 50

    # Different classifiers in testing, used in cross-dataset evaluation
    _C.MODEL.RESET_CLS_TESTS = False
    _C.MODEL.TEST_CLASSIFIERS = []
    _C.MODEL.TEST_NUM_CLASSES = []

    # Backbones
    _C.MODEL.SWIN = CN()
    _C.MODEL.SWIN.SIZE = "T"  # 'T', 'S', 'B'
    _C.MODEL.SWIN.USE_CHECKPOINT = False
    _C.MODEL.SWIN.OUT_FEATURES = (1, 2, 3)  # FPN stride 8 - 32

    _C.MODEL.TIMM = CN()
    _C.MODEL.TIMM.BASE_NAME = "resnet50"
    _C.MODEL.TIMM.OUT_LEVELS = (3, 4, 5)
    _C.MODEL.TIMM.NORM = "FrozenBN"
    _C.MODEL.TIMM.FREEZE_AT = 0
    _C.MODEL.TIMM.PRETRAINED = False
    _C.MODEL.DATASET_LOSS_WEIGHT = []

    # Multi-dataset dataloader
    _C.DATALOADER.DATASET_RATIO = [1, 1]  # sample ratio
    _C.DATALOADER.USE_RFS = [False, False]
    _C.DATALOADER.MULTI_DATASET_GROUPING = (
        False  # Always true when multi-dataset is enabled
    )
    _C.DATALOADER.DATASET_ANN = ["box", "box"]  # Annotation type of each dataset
    _C.DATALOADER.USE_DIFF_BS_SIZE = False  # Use different batchsize for each dataset
    _C.DATALOADER.DATASET_BS = [8, 32]  # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.DATASET_INPUT_SIZE = [896, 384]  # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.DATASET_INPUT_SCALE = [
        (0.1, 2.0),
        (0.5, 1.5),
    ]  # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.DATASET_MIN_SIZES = [
        (640, 800),
        (320, 400),
    ]  # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.DATASET_MAX_SIZES = [1333, 667]  # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.USE_TAR_DATASET = (
        False  # for ImageNet-21K, directly reading from unziped files
    )
    _C.DATALOADER.TARFILE_PATH = "datasets/imagenet/metadata-22k/tar_files.npy"
    _C.DATALOADER.TAR_INDEX_DIR = "datasets/imagenet/metadata-22k/tarindex_npy"

    _C.SOLVER.USE_CUSTOM_SOLVER = False
    _C.SOLVER.OPTIMIZER = "SGD"
    _C.SOLVER.BACKBONE_MULTIPLIER = 1.0  # Used in DETR
    _C.SOLVER.CUSTOM_MULTIPLIER = 1.0  # Used in DETR
    _C.SOLVER.CUSTOM_MULTIPLIER_NAME = []  # Used in DETR
    _C.SOLVER.TRAIN_ITER = -1

    _C.INPUT.CUSTOM_AUG = ""
    _C.INPUT.TRAIN_SIZE = 640
    _C.INPUT.TEST_SIZE = 640
    _C.INPUT.SCALE_RANGE = (0.1, 2.0)
    # 'default' for fixed short/ long edge, 'square' for max size=INPUT.SIZE
    _C.INPUT.TEST_INPUT_TYPE = "default"

    _C.FIND_UNUSED_PARAM = True
    _C.EVAL_PRED_AR = False
    _C.EVAL_PROPOSAL_AR = False
    _C.EVAL_CAT_SPEC_AR = False
    _C.DEBUG = False
    _C.SAVE_DEBUG = False
    _C.IS_DEBUG = False
    _C.QUICK_DEBUG = False
    _C.FP16 = False
    _C.EVAL_AP_FIX = False
    _C.GEN_PSEDO_LABELS = False
    _C.SAVE_DEBUG_PATH = "output/save_debug/"

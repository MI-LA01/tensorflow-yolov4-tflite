#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()
# Set the class name
__C.YOLO.CLASSES              = "./data/classes/hails.names"
__C.YOLO.ANCHORS              = "./data/anchors/basline_anchors.txt"
__C.YOLO.ANCHORS_TINY         = "./data/anchors/basline_tiny_anchors.txt"
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.STRIDES_TINY         = [16, 32]
__C.YOLO.XYSCALE              = [1.2, 1.1, 1.05]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.6


# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "/content/drive/e4-export-images-for-ai/output/object_detection/hails/yolo/hails_train.txt"
__C.TRAIN.BATCH_SIZE          = 16
__C.TRAIN.INPUT_SIZE          = 320
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-5
__C.TRAIN.WARMUP_EPOCHS       = 6
__C.TRAIN.FISRT_STAGE_EPOCHS    = 0
__C.TRAIN.SECOND_STAGE_EPOCHS   = 30



# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "/content/drive/e4-export-images-for-ai/output/object_detection/hails/yolo/hails_validation.txt"
__C.TEST.BATCH_SIZE           = 1
__C.TEST.INPUT_SIZE           = 320
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.4
__C.TEST.IOU_THRESHOLD        = 0.5



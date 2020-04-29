import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov4 import YOLOv4, decode
from PIL import Image
from matplotlib.pyplot import imshow
from urllib.request import urlopen
from scipy.misc import imread

INPUT_SIZE   = 320
NUM_CLASS    = len(utils.read_class_names(cfg.YOLO.CLASSES))
CLASSES      = utils.read_class_names(cfg.YOLO.CLASSES)

print(NUM_CLASS)
print(CLASSES)

# Build Model
input_layer  = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
feature_maps = YOLOv4(input_layer)

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)
    print(bbox_tensors)

model = tf.keras.Model(input_layer, bbox_tensors)
model.load_weights("./checkpoints/yolov4")
model.summary()
model.save("./e4.h5")

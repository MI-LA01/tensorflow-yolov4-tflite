from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
import sys
from core.config import cfg
from core.yolov4 import YOLOv4, YOLOv3, decode

flags.DEFINE_string('weights', '/media/user/Source/Code/VNPT/cv_models_quantization/tflite/yolov3_tflite/data/yolov3.weights',
                    'path to weights file')
flags.DEFINE_string('framework', 'tf', 'select model type in (tf, tflite)'
                    'path to weights file')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 512, 'resize images to')
flags.DEFINE_string('annotation_path', "./data/dataset/val2017.txt", 'annotation path')
flags.DEFINE_string('write_image_path', "./data/detection/", 'write image path')
flags.DEFINE_boolean('tensorboard', True, 'report in tensorbard the predicted images?')
flags.DEFINE_string('log', './data/log', 'default tensorflow log file')
flags.DEFINE_integer('step', 0, 'The current training step')

def main(_argv):
    INPUT_SIZE = FLAGS.size
    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)
    predicted_dir_path = './mAP/predicted'
    ground_truth_dir_path = './mAP/ground-truth'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

    #Used for tensorboard
    images = []
    writer = tf.summary.create_file_writer(FLAGS.log)

    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)

    # Build Model
    if FLAGS.framework == "tf":
        input_layer = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
        
        feature_maps = YOLOv4(input_layer)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = decode(fm, i)
            bbox_tensors.append(bbox_tensor)

        model = tf.keras.Model(input_layer, bbox_tensors)
        
        optimizer   = tf.keras.optimizers.Adam()
        ckpt        = tf.train.Checkpoint(step=tf.Variable(1, trainable=False, dtype=tf.int64), optimizer=optimizer, net=model)
        manager     = tf.train.CheckpointManager(ckpt, './checkpoints', max_to_keep=10)

        ckpt.restore(manager.latest_checkpoint)

        #model.load_weights(ckpt)
   
    num_lines = sum(1 for line in open(FLAGS.annotation_path))

    with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            annotation = line.strip().split()
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

            print('=> ground truth of %s:' % image_name)
            num_bbox_gt = len(bboxes_gt)
            with open(ground_truth_path, 'w') as f:
                for i in range(num_bbox_gt):
                    class_name = CLASSES[classes_gt[i]]
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print('=> predict result of %s:' % image_name)
            predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
            # Predict Process
            image_size = image.shape[:2]
            image_data = utils.image_preporcess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            if FLAGS.framework == "tf":
                pred_bbox = model.predict(image_data)
            else:
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred_bbox = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3':
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)
            elif FLAGS.model == 'yolov4':
                XYSCALE = cfg.YOLO.XYSCALE
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE=XYSCALE)

                xy_grid = np.tile(tf.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
                xy_grid = xy_grid.astype(np.float)

                pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
                #pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
                pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
                pred[:, :, :, :, 0:4] = tf.concat([pred_xy, pred_wh], axis=-1)
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)
            bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
            bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')

            image = utils.draw_bbox(image, bboxes)
            images.append(image)

            # if cfg.TEST.DECTECTED_IMAGE_PATH is not None:
            #     image = utils.draw_bbox(image, bboxes)
            #     #cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH + image_name, image)
            #     images.append(image)

            # with open(predict_result_path, 'w') as f:
            #     for bbox in bboxes:
            #         coor = np.array(bbox[:4], dtype=np.int32)
            #         score = bbox[4]
            #         class_ind = int(bbox[5])
            #         class_name = CLASSES[class_ind]
            #         score = '%.4f' % score
            #         xmin, ymin, xmax, ymax = list(map(str, coor))
            #         bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
            #         f.write(bbox_mess)
            #         print('\t' + str(bbox_mess).strip())
            # print(num, num_lines)
            
    with writer.as_default():
        images_2 = np.reshape(images[0:1000], (-1, 320, 320, 3))
        tf.summary.image("image data examples", images_2, max_outputs=1000, step=FLAGS.step)

if __name__ == '__main__':
    try:
        app.run(main)
        sys.exit()
    except SystemExit:
        pass



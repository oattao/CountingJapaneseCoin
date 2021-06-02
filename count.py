import os
import argparse
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from utils.retina import RetinaNet, PredictionDecoder
from utils.image import resize_and_pad_image, visualize_detections, prepare_image

class_dict = {0.0: '100 yen', 1.0: '10 yen', 2.0: 'unknown'}

# parse input parameter
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir')
parser.add_argument('--image_dir')
args = parser.parse_args()
model_dir = args.model_dir
image_dir = args.image_dir

# load pretrain model
model = RetinaNet(3)
latest_checkpoint = tf.train.latest_checkpoint(model_dir)
model.load_weights(latest_checkpoint)

# make inference model
image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = PredictionDecoder(num_classes=4, confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

# load and prepare image
image = cv.imread(image_dir)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
input_image, ratio = prepare_image(image)

# predict ans show result
detections = inference_model.predict(input_image)
num_detections = detections.valid_detections[0]
# breakpoint()
class_names = [class_dict[pred] for pred in detections.nmsed_classes[0][:num_detections]]
visualize_detections(image, detections.nmsed_boxes[0][:num_detections] / ratio,
                     class_names, detections.nmsed_scores[0][:num_detections])
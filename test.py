import os
import argparse
from functools import partial
import tensorflow as tf
from tensorflow.data import TextLineDataset
from tensorflow.keras.models import load_model, Model
from utils.retina import RetinaNet, PredictionDecoder
from utils.generator import preprocess_data_from_textline
from utils.image import resize_and_pad_image, visualize_detections, prepare_image

# parse input parameter
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir')
args = parser.parse_args()
model_dir = args.model_dir

val_dataset  = TextLineDataset(filenames='data/test_annotation.txt')
val_path = "data/images"
# val_dataset = val_dataset.shuffle(800)
val_dataset = val_dataset.map(lambda x: preprocess_data_from_textline(x, val_path))
class_dict = {0.0: '100 yen', 1.0: '10 yen', 2.0: 'unknown'}

model = RetinaNet(3)
latest_checkpoint = tf.train.latest_checkpoint(model_dir)
model.load_weights(latest_checkpoint)

image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = PredictionDecoder(num_classes=4, confidence_threshold=0.54)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

for sample in val_dataset.take(8):
    original_image = sample[0]
    input_image, ratio = prepare_image(original_image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [class_dict[pred] for pred in detections.nmsed_classes[0][:num_detections]]
    visualize_detections(original_image, detections.nmsed_boxes[0][:num_detections] / ratio,
                         class_names, detections.nmsed_scores[0][:num_detections])
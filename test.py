from tensorflow.data import TextLineDataset
from functools import partial
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from utils.retina import RetinaNet, PredictionDecoder
from utils.generator import preprocess_data_from_textline
from utils.box import resize_and_pad_image, visualize_detections

val_dataset  = TextLineDataset(filenames='data/test_annotation.txt')
val_path = "data/images"
val_dataset = val_dataset.map(lambda x: preprocess_data_from_textline(x, val_path))

def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio


model = RetinaNet(3)
latest_checkpoint = tf.train.latest_checkpoint('models')
model.load_weights(latest_checkpoint)

image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = PredictionDecoder(num_classes=4, confidence_threshold=0.38)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

for sample in val_dataset.take(4):
    original_image = sample[0]
    input_image, ratio = prepare_image(original_image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = ['100 yen'] * num_detections
    score = [1] * num_detections
    visualize_detections(original_image, detections.nmsed_boxes[0][:num_detections] / ratio,
                         class_names, detections.nmsed_scores[0][:num_detections])
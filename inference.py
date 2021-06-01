import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model, Model
import tensorflow_datasets as tfds

from utils.retina import PredictionDecoder, RetinaNet
from utils.box import prepare_image, visualize_detections

model = RetinaNet(num_classes=80)
latest_checkpoint = tf.train.latest_checkpoint('data')
model.load_weights(latest_checkpoint)

image = Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = PredictionDecoder(num_classes=80, confidence_threshold=0.5)(image, predictions)
inference_model = Model(inputs=image, outputs=detections)




# val_dataset = tfds.load("coco/2017", split="validation", data_dir="data")
(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
)
int2str = dataset_info.features["objects"]["label"].int2str

for sample in val_dataset.take(2):
    image = tf.cast(sample["image"], dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [
        int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]
    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )
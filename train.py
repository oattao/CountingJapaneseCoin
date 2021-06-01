miport os
from utils.retina import RetinaNet, FocalLoss, get_backbone
from utils.generator import (LabelEncoder, preprocess_data, 
                             preprocess_data_from_textline, 
                             preprocess_data_from_tfrecord)
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.data import TextLineDataset, TFRecordDataset
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
# from tensorflow.optimizers.schedules import PiecewiseConstantDecay
import zipfile

import pdb

model_dir = "models"
label_encoder = LabelEncoder()

num_classes = 4

learning_rates = [2.5e-06, 0.000625, 0.001, 0.00125, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries=learning_rate_boundaries, 
                                          values=learning_rates)


# resnet50_backbone = get_backbone()
loss_fn = FocalLoss(num_classes)
# model = RetinaNet(num_classes=num_classes)

model = RetinaNet(4)
latest_checkpoint = tf.train.latest_checkpoint('models/')
model.load_weights(latest_checkpoint)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
# optimizer = Adam(learning_rate=0.001)
model.compile(loss=loss_fn, optimizer=optimizer)

batch_size = 3

# (train_dataset, val_dataset), dataset_info = tfds.load(
#     "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
# )

train_dataset  = TextLineDataset(filenames='data/train_annotation.txt')
val_dataset  = TextLineDataset(filenames='data/test_annotation.txt')
train_path = "data/images"
val_path = "data/images"

autotune = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.map(lambda x: preprocess_data_from_textline(x, train_path),
                                  num_parallel_calls=autotune)
# train_dataset = TFRecordDataset(filenames='data/train.record', num_parallel_reads=autotune)
# train_dataset.map(lambda x: preprocess_data_from_tfrecord(x, feature_description), num_parallel_calls=autotune)

train_dataset = train_dataset.shuffle(800)
train_dataset = train_dataset.padded_batch(batch_size=batch_size, 
                                           padding_values=(0.0, 1e-8, -1), 
                                           drop_remainder=True)
train_dataset = train_dataset.map(label_encoder.encode_batch, 
                                  num_parallel_calls=autotune)
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(autotune)

# val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
val_dataset = val_dataset.map(lambda x: preprocess_data_from_textline(x, val_path),
                              num_parallel_calls=autotune)

# val_dataset = TFRecordDataset(filenames='data/val.record', num_parallel_reads=autotune)
# val_dataset.map(lambda x: preprocess_data_from_tfrecord(x, feature_description), num_parallel_calls=autotune)
val_dataset = val_dataset.padded_batch(batch_size=batch_size, 
                                       padding_values=(0.0, 1e-8, -1), 
                                       drop_remainder=True)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)

# saving_cb = ModelCheckpoint(filepath=os.path.join(model_dir,'retinanet.h5'),
#                             monitor="val_loss",
#                             save_best_only=True,
#                             save_weights_only=True,
#                             verbose=1)

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
]

model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=30,
          callbacks=callbacks_list,
          verbose=1)
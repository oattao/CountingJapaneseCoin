import os
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


# model_dir = "models"
model_dir = 'models_2'
label_encoder = LabelEncoder()

num_classes = 3

learning_rates = [2.5e-06, 0.000625, 0.0005, 0.001, 0.00025, 2.5e-05]
learning_rate_boundaries = [55, 125, 200, 300, 400]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries=learning_rate_boundaries, 
                                          values=learning_rates)


resnet50_backbone = get_backbone()
loss_fn = FocalLoss(num_classes)
model = RetinaNet(num_classes=num_classes)

# model = RetinaNet(4)
# latest_checkpoint = tf.train.latest_checkpoint('models/')
# model.load_weights(latest_checkpoint)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
# optimizer = Adam(learning_rate=0.0005)
model.compile(loss=loss_fn, optimizer=optimizer)

batch_size = 2

train_dataset  = TextLineDataset(filenames='data/train_annotation.txt')
val_dataset  = TextLineDataset(filenames='data/test_annotation.txt')
train_path = "data/images"
val_path = "data/images"

autotune = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.map(lambda x: preprocess_data_from_textline(x, train_path),
                                  num_parallel_calls=autotune)

train_dataset = train_dataset.shuffle(800)
train_dataset = train_dataset.padded_batch(batch_size=batch_size, 
                                           padding_values=(0.0, 1e-8, -1), 
                                           drop_remainder=True)
train_dataset = train_dataset.map(label_encoder.encode_batch, 
                                  num_parallel_calls=autotune)
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(autotune)

val_dataset = val_dataset.map(lambda x: preprocess_data_from_textline(x, val_path),
                              num_parallel_calls=autotune)

val_dataset = val_dataset.padded_batch(batch_size=batch_size, 
                                       padding_values=(0.0, 1e-8, -1), 
                                       drop_remainder=True)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)

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
          epochs=50,
          callbacks=callbacks_list,
          verbose=1)
from tensorflow.keras.layers import Layer, Conv2D, UpSampling2D, ReLU, Input, InputLayer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.initializers import RandomNormal
import tensorflow as tf
from tensorflow.keras.losses import Loss
from utils.box import convert_to_corners
from utils.anchor import AnchorBox

import numpy as np

# Utilites for model
def get_backbone():
	backbone = ResNet50(include_top=False, input_shape=[None, None, 3])
	c3_output = backbone.get_layer('conv3_block4_out').output
	c4_output = backbone.get_layer('conv4_block6_out').output
	c5_output = backbone.get_layer('conv5_block3_out').output

	model = Model(inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output])
	return model

class FeaturePyramid(Layer):
	def __init__(self, backbone=None, **kwargs):
		super(FeaturePyramid, self).__init__(name='FeaturePyramid', **kwargs)
		self.backbone = backbone if backbone else get_backbone()
		self.conv_c3_1x1 = Conv2D(256, 1, 1, 'same')
		self.conv_c4_1x1 = Conv2D(256, 1, 1, 'same')
		self.conv_c5_1x1 = Conv2D(256, 1, 1, 'same')
		
		self.conv_c3_3x3 = Conv2D(256, 3, 1, 'same')
		self.conv_c4_3x3 = Conv2D(256, 3, 1, 'same')
		self.conv_c5_3x3 = Conv2D(256, 3, 1, 'same')

		self.conv_c6_3x3 = Conv2D(256, 3, 2, 'same')
		self.conv_c7_3x3 = Conv2D(256, 3, 2, 'same')
		self.upsample_2x = UpSampling2D(2)
		self.relu = ReLU()

	def call(self, images, training=False):
		c3_output, c4_output, c5_output = self.backbone(images, training=False)

		p3_output = self.conv_c3_1x1(c3_output)
		p4_output = self.conv_c4_1x1(c4_output)
		p5_output = self.conv_c5_1x1(c5_output)

		p4_output += self.upsample_2x(p5_output)
		p3_output += self.upsample_2x(p4_output)

		p3_output = self.conv_c3_3x3(p3_output)
		p4_output = self.conv_c4_3x3(p4_output)
		p5_output = self.conv_c5_3x3(p5_output)

		p6_output = self.conv_c6_3x3(c5_output)
		p7_output = self.conv_c7_3x3(self.relu(p6_output))

		return p3_output, p4_output, p5_output, p6_output, p7_output

def build_head(output_filters, bias_init):
	head = Sequential([InputLayer(input_shape=[None, None, 256])])
	kernel_init = RandomNormal(mean=0.0, stddev=0.01)
	for _ in range(4):
		head.add(Conv2D(256, 3, padding='same', kernel_initializer=kernel_init))
		head.add(ReLU())
	head.add(Conv2D(output_filters, 3, 1, 'same', 
					kernel_initializer=kernel_init,
					bias_initializer=bias_init))	
	return head

class RetinaNet(Model):
	def __init__(self, num_classes, num_anchors=9, backbone=None, **kwargs):
		super(RetinaNet, self).__init__(name='RetinaNet', **kwargs)
		self.fpn = FeaturePyramid(backbone)
		self.num_classes = num_classes
		self.num_anchors = num_anchors
		prior_probability = tf.constant_initializer(-np.log((1-0.01) / 0.01))
		self.cls_head = build_head(self.num_anchors * self.num_classes,
								   prior_probability)
		self.box_head = build_head(self.num_anchors * 4, 'zeros')

	def call(self, image, training=False):
		features = self.fpn(image, training=training)
		N = tf.shape(image)[0] # batch_size
		cls_outputs, box_outputs = [], []
		for feature in features:
			box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
			cls_outputs.append(tf.reshape(self.cls_head(feature), [N, -1, self.num_classes]))

		cls_outputs = tf.concat(cls_outputs, axis=1)
		box_outputs = tf.concat(box_outputs, axis=1)
		outputs = tf.concat([box_outputs, cls_outputs], axis=-1)

		return outputs

class PredictionDecoder(Layer):
	def __init__(self, num_classes, confidence_threshold=0.05,
				 nms_iou_threshold=0.5, max_detections_per_class=100,
				 max_detections=100, box_variance=[0.1, 0.1, 0.2, 0.2], **kwargs):
		super(PredictionDecoder, self).__init__(**kwargs)
		self.num_classes = num_classes
		self.confidence_threshold = confidence_threshold
		self.nms_iou_threshold=nms_iou_threshold
		self.max_detections_per_class=max_detections_per_class
		self.max_detections = max_detections
		self.anchor_box = AnchorBox()
		self.box_variance = tf.convert_to_tensor(box_variance, dtype=tf.float32)

	def decode(self, anchor_boxes, box_predictions):
		boxes = box_predictions * self.box_variance
		boxes = tf.concat([boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
						   tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:]], axis=-1)		
		boxes_transformed = convert_to_corners(boxes)
		return boxes_transformed

	def call(self, images, predictions):
		image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
		anchor_boxes = self.anchor_box.get_anchors(image_shape[1], image_shape[2])
		box_predictions = predictions[:, :, :4]
		cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
		boxes = self.decode(anchor_boxes[None, ...], box_predictions)

		return tf.image.combined_non_max_suppression(
			tf.expand_dims(boxes, axis=2),
			cls_predictions,
			self.max_detections_per_class,
			self.max_detections,
			self.nms_iou_threshold,
			self.confidence_threshold,
			clip_boxes=False)

# Focal loss:
class BoxLoss(Loss):
	def __init__(self, delta):
		super(BoxLoss, self).__init__(reduction='none', name='BoxLoss')
		self.delta = delta

	def call(self, y_true, y_pred):
		difference = y_true - y_pred
		absolute_difference = tf.abs(difference)
		squared_difference = tf.square(difference)
		loss = tf.where(tf.less(absolute_difference, self.delta),
						0.5 * squared_difference, absolute_difference - 0.5)
		return tf.reduce_sum(loss, axis=-1)

class ClassLoss(Loss):
	def __init__(self, alpha, gamma):
		super(ClassLoss, self).__init__(reduction='none', name='ClassLoss')
		self.alpha = alpha
		self.gamma = gamma

	def call(self, y_true, y_pred):
		cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
		probs = tf.nn.sigmoid(y_pred)
		alpha = tf.where(tf.equal(y_true, 1.0), self.alpha, (1.0 - self.alpha))
		pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
		loss = alpha * tf.pow(1.0 - pt, self.gamma) * cross_entropy
		return tf.reduce_sum(loss, axis=-1)

class FocalLoss(Loss):
	def __init__(self, num_classes, alpha=0.25, gamma=2.0, delta=1.0):
		super(FocalLoss, self).__init__(reduction='auto', name='FocalLoss')
		self.clf_loss = ClassLoss(alpha, gamma)
		self.box_loss = BoxLoss(delta)
		self.num_classes = num_classes

	def call(self, y_true, y_pred):
		y_pred = tf.cast(y_pred, dtype=tf.float32)
		box_labels = y_true[:, :, :4]
		box_predictions = y_pred[:, :, :4]
		cls_labels = tf.one_hot(tf.cast(y_true[:, :, 4], dtype=tf.int32), 
								depth=self.num_classes, dtype=tf.float32)
		cls_predictions = y_pred[:, :, 4:]

		positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
		ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)		
		cls_loss = self.clf_loss(cls_labels, cls_predictions)		
		box_loss = self.box_loss(box_labels, box_predictions)
		cls_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, cls_loss)
		box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
		normalizer = tf.reduce_sum(positive_mask, axis=-1)
		cls_loss = tf.math.divide_no_nan(tf.reduce_sum(cls_loss, axis=-1), normalizer)
		box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
		loss = cls_loss + box_loss
		return loss		
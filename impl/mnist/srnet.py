import matplotlib.pyplot as plt
import numpy as np
import time
import functools
import os
import tensorflow as tf
import cv2

def load_batch(dataset_path, dataset_name, split_name, batch_size=128, image_size=[64, 64, 3]):
	#1. Data pipeline
	dataset = mnist_reader.get_split(split_name, dataset_path)
	print(dataset_name)
	print(split_name)
	data_provider = slim.dataset_data_provider.DatasetDataProvider(
		            dataset, common_queue_capacity=4*batch_size, common_queue_min=batch_size)    
	[image, label] = data_provider.get(['image', 'label'])

	image = (tf.to_float(image) - 128.0) / 128.0 # convert 0~255 scale into -1~1 scale
	image.set_shape(image_size)
	images, labels = tf.train.batch(
		      [image, label],
		      batch_size=batch_size,
		      num_threads=4,
		      capacity=2 * batch_size)
	return dataset, images, labels



def train(checkpoint_path, dataset_path, batch_size, result_path, weight):
	
import matplotlib.pyplot as plt
import numpy as np
import time
import functools
import os
import tensorflow as tf
import cv2
from model.mnist import srnet
from tensorflow.python.ops import variable_scope



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

def _convert_tensor_or_l_or_d(tensor_or_l_or_d):
  """Convert input, list of inputs, or dictionary of inputs to Tensors."""
  if isinstance(tensor_or_l_or_d, (list, tuple)):
    return [ops.convert_to_tensor(x) for x in tensor_or_l_or_d]
  elif isinstance(tensor_or_l_or_d, dict):
    return {k: ops.convert_to_tensor(v) for k, v in tensor_or_l_or_d.items()}
  else:
    return ops.convert_to_tensor(tensor_or_l_or_d)


def train(checkpoint_path, dataset_path, batch_size, result_path, weight):
	with tf.Graph().as_default():
		
		#1. Data pipeline
		dataset, images, labels = load_batch(dataset_path, 'mnist', 'train', batch_size=batch_size, image_size=image_size)
		one_hot_labels = tf.one_hot(labels, dataset.num_classes)
			
		#Todo : take images for visual feature information
		#complete!

		visual_feature = {**feature_list['discrete'], **feature_list['continuous']}
		visual_feature_path = '/home/dan/prj/lab/datasets/visual_feature_samples_multinumber'
		visual_feature_images = {}

		for key in visual_feature.keys():
			visual_feature_images[key] = {}
			for attribute in visual_feature[key]:
				visual_feature_images[key][attribute] = []
				path = os.path.join(visual_feature_path, key, str(attribute))
				for img in os.listdir(path):
					sample = cv2.imread(os.path.join(path, img))
					sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
					sample = (tf.to_float(sample) - 128.0) / 128.0
					sample = tf.reshape(sample, image_size)
					visual_feature_images[key][attribute].append(sample)


		#2. model deploy
		
		#input
		#noise, disentangled_rep

		noise = _convert_tensor_or_l_or_d(noise)
		disentangled_rep = _convert_tensor_or_l_or_d(disentangled_rep)
		generator_inputs = tf.concat([noise, disentangled_rep], axis = 1)

		with variable_scope.variable_scope('generator') as gen_scope:
			generated_data = srnet.generator(generator_inputs)

		with variable_scope.variable_scope('discriminator') as dis_scope:
			dis_gen_data = srnet.discriminator(generated_data, generator_inputs)

		with variable_scope.variable_scope(dis_scope, reuse = True):
			real_data = ops.convert_to_tensor(real_data)
			dis_real_data = srnet.discriminator(real_data, generator_inputs)

		with variable_scope.variable_scope('encoder') as encoder:
			semantic_rep = srnet.encoder()
		with variable_scope.variable_scope('decoder') as decoder:
			code = srnet.decoder(semantic_rep)



		if check_shapes:
			if not generated_data.shape.is_compatible_with(real_data.shape):
				raise ValueError(
					'Generator output shape (%s) must be the same shape as real data '
					'(%s).' % (generated_data.shape, real_data.shape))




		#2. model deploy

		# Dimensions of the structured and unstructured noise dimensions.
		#todo: megan_model need to be designed. It must have the part of visual feature check.
		#complete : the part of tracing variant of visual feature is implemented. 
		cat_dim, cont_dim, noise_dims = 10, 2, 64
		unstructured_noise_dims = noise_dims - cont_dim

		generator_fn = functools.partial(gan_networks.generator, categorical_dim=cat_dim)
		discriminator_fn = functools.partial(
		    gan_networks.discriminator, categorical_dim=cat_dim,
		    continuous_dim=cont_dim)
		unstructured_inputs, structured_inputs = gan_networks.get_infogan_noise(
		    batch_size, cat_dim, cont_dim, noise_dims)

		megan_model = gan_train.megan_model(
		    generator_fn=generator_fn,
		    discriminator_fn=discriminator_fn,
		    real_data=images,
		    visual_feature_images = visual_feature_images,
		    feature_list = feature_list,
		    unstructured_generator_inputs=unstructured_inputs,
		    structured_generator_inputs=structured_inputs
		    )

		# from tensorflow.python.ops import math_ops
		# print(dir(megan_model))

		# with tf.Session() as sess:
		# 	sess.run(tf.global_variables_initializer())
		# 	with slim.queues.QueueRunners(sess):
		# 		print(sess.run(math_ops.to_float(megan_model.discriminator_gen_outputs)))

			

		#Todo : I need to design loss function for megan
		#3. training op
		megan_loss = gan_train.gan_loss(
		    megan_model,
		    gradient_penalty_weight=1.0,
		    mutual_information_penalty_weight=1.0,
		    visual_feature_regularizer_weight=weight)

		# Sanity check that we can evaluate our losses.
		visual_gan.evaluate_tfgan_loss(megan_loss)
		generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
		discriminator_optimizer = tf.train.AdamOptimizer(0.00009, beta1=0.5)
		gan_train_ops = gan_train.gan_train_ops(
		    megan_model,
		    megan_loss,
		    generator_optimizer,
		    discriminator_optimizer)

		#4. Session run learning op

		global_step = tf.train.get_or_create_global_step()
		train_step_fn = gan_train.get_sequential_train_steps()
		loss_values, mnist_score_values  = [], []
		saver = tf.train.Saver()


		summary_op = tf.summary.merge_all()
		print(dir(summary_op))
		train_writer = tf.summary.FileWriter(checkpoint_path + '/train')


		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			with slim.queues.QueueRunners(sess):
				start_time = time.time()
				for i in range(100000):
					cur_loss, _ = train_step_fn(
					sess, gan_train_ops, global_step, train_step_kwargs={})
					loss_values.append((i, cur_loss))
					if i % 500 == 0:
						visual_gan.varying_categorical_noise(sess, megan_model, 10, unstructured_noise_dims, cont_dim, i, result_path)
						visual_gan.varying_noise_continuous_ndim(sess, megan_model, 10, 0, unstructured_noise_dims, cont_dim
	    																,i, result_path)
						visual_gan.varying_noise_continuous_ndim(sess, megan_model, 10, 1, unstructured_noise_dims, cont_dim
	    																,i, result_path)
						train_writer.add_summary(sess.run(summary_op), i)
					if i % 1000 == 0: 
						print('Current loss: %f' % cur_loss)
						if not tf.gfile.Exists(checkpoint_path):
							tf.gfile.MakeDirs(checkpoint_path)
						save_dir = os.path.join(checkpoint_path, "megan"+'_'+str(i)+'.ckpt')
						saver.save(sess, save_dir)
						print("Model saved in file: %s" % checkpoint_path)

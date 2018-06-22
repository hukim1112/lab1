import tensorflow as tf
slim = tf.contrib.slim
tfgan = tf.contrib.gan
layers = tf.contrib.layers
ds = tf.contrib.distributions
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import ops
import numpy as np
import visualizations, losses_fn
from datasets.reader import mnist as mnist_reader


leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

def get_noise(batch_size, total_continuous_noise_dims):
  """Get unstructured and structured noise for InfoGAN.

  Args:
    batch_size: The number of noise vectors to generate.
    categorical_dim: The number of categories in the categorical noise.
    structured_continuous_dim: The number of dimensions of the uniform
      continuous noise.
    total_continuous_noise_dims: The number of continuous noise dimensions. This
      number includes the structured and unstructured noise.

  Returns:
    A 2-tuple of structured and unstructured noise. First element is the
    unstructured noise, and the second is a 2-tuple of
    (categorical structured noise, continuous structured noise).
  """
  # Get unstructurd noise.
  noise = tf.random_normal(
      [batch_size, total_continuous_noise_dims])

  return noise




class Gan():
    def __init__(self, data):

        self.graph = tf.Graph()
        self.sess = tf.Session()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(graph = self.graph, config=tf.ConfigProto(gpu_options=gpu_options))

        self.generator = generator
        self.discriminator = discriminator

        self.data = data

        # data
        self.cat_dim = self.data.cat_dim
        self.code_con_dim = self.data.code_con_dim
        self.total_con_dim = self.data.total_con_dim
        self.channel = self.data.channel
        self.dataset_path = self.data.path
        self.dataset_name = self.data.name
        self.split_name = self.data.split_name
        self.batch_size = self.data.batch_size

        with self.graph.as_default():
            with slim.queues.QueueRunners(self.sess):
                self.initializer = tf.global_variables_initializer()
                self.dataset, self.real_data, self.labels = load_batch(self.dataset_path, self.dataset_name, self.split_name, self.batch_size)
                tf.train.start_queue_runners(self.sess)               
                self.gen_input_noise = get_noise(self.batch_size, self.total_con_dim)

                #if this model done well, erase it
                # self.real_data = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
                # self.gen_input_noise = tf.placeholder(tf.float32, shape=[None, self.z_dim])
                # self.gen_input_code = tf.placeholder(tf.float32, shape=[None, 2])

                with variable_scope.variable_scope('generator') as self.gen_scope:
                    self.gen_data = self.generator(self.gen_input_noise) #real/fake loss
                
                with variable_scope.variable_scope('discriminator') as self.dis_scope:
                    self.dis_gen_data = self.discriminator(self.gen_data) #real/fake loss + I(c' ; X_{data}) loss
                with variable_scope.variable_scope(self.dis_scope, reuse = True):
                    self.real_data = ops.convert_to_tensor(self.real_data)
                    self.dis_real_data = self.discriminator(self.real_data) #real/fake loss 
                print(self.dis_scope.name)
                #TO do code loss functions.
                #loss
                self.dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.dis_scope.name)
                self.gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.gen_scope.name)

                self.D_loss = losses_fn.wasserstein_discriminator_loss(self.dis_real_data, self.dis_gen_data)
                self.G_loss = losses_fn.wasserstein_generator_loss(self.dis_gen_data)
                #self.wasserstein_gradient_penalty_loss = losses.wasserstein_gradient_penalty(what?)

                #solver
                self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.dis_var)
                self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.gen_var)
     
                self.saver = tf.train.Saver()

    def train(self, result_dir, ckpt_dir, training_iteration = 1000000):
        # Make this train from the latest checkpoint!
        path_to_latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if path_to_latest_ckpt == None:
            self.sess.run(self.initializer)
        else:
            self.saver.restore(self.sess, path_to_latest_ckpt)

        for i in range(training_iteration):
            #if this model done well, erase it
            # dataset, real_images, labels = load_batch(self.dataset_path, self.dataset_name, self.split_name, self.batch_size)
            # gen_input_noise, gen_input_code = get_infogan_noise(self.batch_size, self.cat_dim, self.structured_con_dim, self.total_con_dim)

            for _ in range(1):
                self.sess.run(self.D_solver)
            for _ in range(1):
                self.sess.run(self.G_solver)

            if ((i % 1000) == 0):
            	visualizations.varying_noise_continuous_ndim_without_category(self, order, total_continuous_dim, i, result_dir)



def load_batch(dataset_path, dataset_name, split_name, batch_size=128):

    #1. Data pipeline
    dataset = mnist_reader.get_split(split_name, dataset_path)
    print(dataset_name)
    print(split_name)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset, common_queue_capacity=4*batch_size, common_queue_min=batch_size)    
    [image, label] = data_provider.get(['image', 'label'])
    image = (tf.to_float(image) - 128.0) / 128.0 # convert 0~255 scale into -1~1 scale
    images, labels = tf.train.batch(
              [image, label],
              batch_size=batch_size,
              num_threads=4,
              capacity=2 * batch_size)
    print('batch image size :', images.shape)
    return dataset, images, labels


def generator(gen_input_noise, weight_decay=2.5e-5):
    """InfoGAN discriminator network on MNIST digits.
    
    Based on a paper https://arxiv.org/abs/1606.03657 and their code
    https://github.com/openai/InfoGAN.
    
    Returns:
        A generated image in the range [-1, 1].
    """

    print('noise shape : ', gen_input_noise.shape)
    with slim.arg_scope(
        [layers.fully_connected, layers.conv2d_transpose],
        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.fully_connected(gen_input_noise, 1024)
        net = layers.fully_connected(net, 7 * 7 * 128)
        net = tf.reshape(net, [-1, 7, 7, 128])
        net = layers.conv2d_transpose(net, 64, [3, 3], stride=2)
        net = layers.conv2d_transpose(net, 32, [3, 3], stride=2)
        # Make sure that generator output is in the same range as `inputs`
        # ie [-1, 1].
        net = layers.conv2d(net, 1, 4, normalizer_fn=None, activation_fn=tf.tanh)
    
        return net


def discriminator(img, weight_decay=2.5e-5, categorical_dim=10, continuous_dim=2):
    """InfoGAN discriminator network on MNIST digits.
    
    Based on a paper https://arxiv.org/abs/1606.03657 and their code
    https://github.com/openai/InfoGAN.
    
    Args:
        img: Real or generated MNIST digits. Should be in the range [-1, 1].
        unused_conditioning: The TFGAN API can help with conditional GANs, which
            would require extra `condition` information to both the generator and the
            discriminator. Since this example is not conditional, we do not use this
            argument.
        weight_decay: The L2 weight decay.
        categorical_dim: Dimensions of the incompressible categorical noise.
        continuous_dim: Dimensions of the incompressible continuous noise.
    
    Returns:
        Logits for the probability that the image is real, and a list of posterior
        distributions for each of the noise vectors.
    """
    with slim.arg_scope(
        [layers.conv2d, layers.fully_connected],
        activation_fn=leaky_relu, normalizer_fn=layers.batch_norm,
        weights_regularizer=layers.l2_regularizer(weight_decay),
        biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [3, 3], stride=2)
        net = layers.conv2d(net, 128, [3, 3], stride=2)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)
    
        logits_real = layers.fully_connected(net, 1, activation_fn=None)

        return logits_real
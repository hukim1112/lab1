import tensorflow as tf
slim = tf.contrib.slim
tfgan = tf.contrib.gan
layers = tf.contrib.layers
ds = tf.contrib.distributions
import numpy as np


leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

def get_infogan_noise(batch_size, categorical_dim, structured_continuous_dim,
                      total_continuous_noise_dims):
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
  unstructured_noise = tf.random_normal(
      [batch_size, total_continuous_noise_dims - structured_continuous_dim])

  # Get categorical noise Tensor.
  categorical_dist = ds.Categorical(logits=tf.zeros([categorical_dim]))
  categorical_noise = categorical_dist.sample([batch_size])
  categorical_noise = tf.one_hot(categorical_noise, categorical_dim)

  # Get continuous noise Tensor.
  continuous_dist = ds.Uniform(-tf.ones([structured_continuous_dim]),
                               tf.ones([structured_continuous_dim]))
  continuous_noise = continuous_dist.sample([batch_size])

  return [unstructured_noise], [categorical_noise, continuous_noise]


class Srnet():
    def __init__(self):

        self.graph = tf.Graph()
        self.sess = tf.Session()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder
        self.decoder = decoder

        self.data = data

        # data
        self.z_dim = self.data.z_dim
        self.c_dim = self.data.y_dim # condition
        self.size = self.data.size
        self.channel = self.data.channel
        self.batch_size = self.data.batch_size

        with graph.as_default():

            self.real_data = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
            self.gen_input_noise = tf.placeholder(tf.float32, shape=[None, self.z_dim])
            self.gen_input_code = tf.placeholder(tf.float32, shape=[None, self.c_dim])

            with variable_scope.variable_scope('generator') as gen_scope:
                gen_data = self.generator(gen_input_noise, gen_input_code) #real/fake loss
            
            with variable_scope.variable_scope('discriminator') as dis_scope:
                dis_gen_data, Q_net = self.discriminator(gen_data) #real/fake loss + I(c' ; X_{data}) loss
            with variable_scope.variable_scope(dis_scope, reuse = True):
                real_data = ops.convert_to_tensor(real_data)
                dis_real_data, _ = self.discriminator(real_data) #real/fake loss 
            with variable_scope.variable_scope(dis_scope, reuse = True):
                visual_feature_tensors = ops.convert_to_tensor(visual_feature_images)
                not_use, visual_feature_code = self.discriminator(visual_feature_tensors, gen_inputs) 
            
            with variable_scope.variable_scope('encoder') as en_scope:
                visual_feature_semantic_rep = self.encoder(visual_feature_code) # Variance-bias Loss
            with variable_scope.variable_scope('decoder') as de_scope:
                reconstructed_code = self.decoder(visual_feature_semantic_rep) #L2(c', c'') reconstruction loss

            with variable_scope.variable_scope(en_scope, reuse = True):
                gen_data_semantic_rep = self.encoder(Q_net)
            with variable_scope.variable_scope(de_scope, reuse = True):
                gen_data_decoded_code = self.decoder(gen_data_semantic_rep)

            #TO do code loss functions.
            #loss
            self.dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dis_scope)
            self.gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=gen_scope)
            self.encoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=en_scope)
            self.decoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=de_scope)

            self.D_loss = losses.wasserstein_discriminator(dis_gen_data, dis_real_data)
            self.G_loss = losses.wasserstein_generator(dis_gen_data)
            #self.wasserstein_gradient_penalty_loss = losses.wasserstein_gradient_penalty(what?)

            self.mutual_information_loss = losses.mutual_information_penalty_weight(gen_inputs, Q_net)

            self.reconstruction_loss1 = losses.mean_square_loss(visual_feature_semantic_rep, reconstructed_code)
            self.reconstruction_loss2 = losses.mean_square_loss(gen_data_semantic_rep, gen_data_decoded_code)
            self.variance_bias_loss = losses.variance_bias_loss(visual_feature_semantic_rep)
            self.total_network_loss = losses.mean_square_loss(gen_input_code, gen_data_decoded_code)

            #solver
            self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.dis_var)
            self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.gen_var)
            self.mutual_information_solver = tf.train.AdamOptimizer().minimize(self.mutual_information_loss, var_list=self.gen_var + self.dis_var)

            self.autoencoder_solver = tf.train.AdamOptimizer().minimize(self.reconstruction_loss1 + self.reconstruction_loss2, var_list=self.encoder_var+self.decoder_var)
            self.semantic_encoder_solver = tf.train.AdamOptimizer().minimize(self.variance_bias_loss, var_list=self.encoder_var + self.dis_var)
            self.total_network_solver = tf.train.AdamOptimizer().minimize(self.total_network_loss, var_list = self.gen_var + self.dis_var + self.encoder_var + self.decoder_var)

    def train(self, sample_dir, ckpt_dir='ckpt', training_iteration = 1000000, batch_size = 64):

        # Make this train from the latest checkpoint!
        i = 0
        self.sess.run(tf.global_variables_initializer())

        for i in range(training_iteration):


    def train(self, sample_dir, ckpt_dir='ckpt', training_epoches = 1000000, batch_size = 64):
        fig_count = 0
        self.sess.run(tf.global_variables_initializer())
        
        for epoch in range(training_epoches):
            X_b, _= self.data(batch_size)
            z_b = sample_z(batch_size, self.z_dim)
            c_b = sample_c(batch_size, self.c_dim)
            # update D
            self.sess.run(
                self.D_solver,
                feed_dict={self.X: X_b, self.z: z_b, self.c: c_b}
                )
            # update G
            for _ in range(1):
                self.sess.run(
                    self.G_solver,
                    feed_dict={self.z: z_b, self.c: c_b}
                )
            # update Q
            for _ in range(2):  
                self.sess.run(
                    self.Q_solver,
                    feed_dict={self.z: z_b, self.c: c_b}
                )
            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                D_loss_curr = self.sess.run(
                        self.D_loss,
                        feed_dict={self.X: X_b, self.z: z_b, self.c: c_b})
                G_loss_curr, Q_loss_curr = self.sess.run(
                        [self.G_loss, self.Q_loss],
                        feed_dict={self.z: z_b, self.c: c_b})
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; Q_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr, Q_loss_curr))

                if epoch % 1000 == 0:
                    z_s = sample_z(16, self.z_dim)
                    c_s = sample_c(16, self.c_dim, fig_count%10)
                    samples = self.sess.run(self.G_sample, feed_dict={self.c: c_s, self.z: z_s})

                    fig = self.data.data2fig(samples)
                    plt.savefig('{}/{}_{}.png'.format(sample_dir, str(fig_count).zfill(3), str(fig_count%10)), bbox_inches='tight')
                    fig_count += 1
                    plt.close(fig)

                #if epoch % 2000 == 0:
                #   self.saver.save(self.sess, os.path.join(ckpt_dir, "infogan.ckpt"))

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


def generator(gen_input_noise, gen_input_code, weight_decay=2.5e-5):
    """InfoGAN discriminator network on MNIST digits.
    
    Based on a paper https://arxiv.org/abs/1606.03657 and their code
    https://github.com/openai/InfoGAN.
    
    Returns:
        A generated image in the range [-1, 1].
    """
    all_noise = tf.concat([gen_input_noise, gen_input_code], axis=1)
    
    with slim.arg_scope(
        [layers.fully_connected, layers.conv2d_transpose],
        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.fully_connected(all_noise, 1024)
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

        # Recognition network for latent variables has an additional layer
        encoder = layers.fully_connected(net, 128, normalizer_fn=layers.batch_norm)

        # Compute logits for each category of categorical latent.
        q_cat = layers.fully_connected(
            encoder, categorical_dim, activation_fn=None)
        #q_cat = ds.Categorical(q_cat)

        # Compute mean for Gaussian posterior of continuous latents.
        q_cont = layers.fully_connected(
            encoder, continuous_dim, activation_fn=None)
        #sigma_cont = tf.ones_like(q_cont)
        #q_cont = ds.Normal(loc=q_cont, scale=sigma_cont)

        return logits_real, [q_cat, q_cont]

def encoder(input, weight_decay=2.5e-3, semantic_dim=2):
    with slim.arg_scope(
        [layers.fully_connected],
        activation_fn=leaky_relu, normalizer_fn=None,
        weights_regularizer=layers.l2_regularizer(weight_decay),
        biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.fully_connected(input, 32, normalizer_fn=layers.batch_norm)
        net = layers.fully_connected(net, 16, normalizer_fn=layers.batch_norm)
        semantic_rep = layers.fully_connected(net, semantic_dim, normalizer_fn=None)
        return semantic_rep

def decoder(input, weight_decay=2.5e-3, continuous_dim = 10):
    with slim.arg_scope(
        [layers.fully_connected],
        activation_fn=leaky_relu, normalizer_fn=None,
        weights_regularizer=layers.l2_regularizer(weight_decay),
        biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.fully_connected(input, 32, normalizer_fn=layers.batch_norm)
        net = layers.fully_connected(net, 16, normalizer_fn=layers.batch_norm)
        disentangled_rep = layers.fully_connected(net, continuous_dim, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
        return disentangled_rep

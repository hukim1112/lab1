import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import variables as contrib_variables_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.distributions import distribution as ds
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops.losses import util
from tensorflow.python.summary import summary


def wasserstein_generator_loss(
    discriminator_gen_outputs,
    weights=1.0,
    scope=None,
    add_summaries=False):
  """Wasserstein generator loss for GANs.

  See `Wasserstein GAN` (https://arxiv.org/abs/1701.07875) for more details.

  Args:
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_gen_outputs`, and must be broadcastable to
      `discriminator_gen_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add detailed summaries for the loss.

  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  discriminator_gen_outputs = math_ops.to_float(discriminator_gen_outputs)
  loss = - discriminator_gen_outputs
  loss = losses.compute_weighted_loss(loss, weights, scope)
  return loss

def wasserstein_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Wasserstein discriminator loss for GANs.

  See `Wasserstein GAN` (https://arxiv.org/abs/1701.07875) for more details.

  Args:
    discriminator_real_outputs: Discriminator output on real data.
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_real_outputs`, and must be broadcastable to
      `discriminator_real_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    generated_weights: Same as `real_weights`, but for
      `discriminator_gen_outputs`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.

  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  discriminator_real_outputs = math_ops.to_float(discriminator_real_outputs)
  discriminator_gen_outputs = math_ops.to_float(discriminator_gen_outputs)
  discriminator_real_outputs.shape.assert_is_compatible_with(discriminator_gen_outputs.shape)

  loss_on_generated = losses.compute_weighted_loss(discriminator_gen_outputs, generated_weights, scope)
  loss_on_real = losses.compute_weighted_loss(discriminator_real_outputs, real_weights, scope)
  loss = loss_on_generated - loss_on_real
  return loss

def wasserstein_gradient_penalty(
    self,
    real_data,
    generated_data,
    epsilon=1e-10,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """The gradient penalty for the Wasserstein discriminator loss.

  See `Improved Training of Wasserstein GANs`
  (https://arxiv.org/abs/1704.00028) for more details.

  Args:
    real_data: Real data.
    generated_data: Output of the generator.
    generator_inputs: Exact argument to pass to the generator, which is used
      as optional conditioning to the discriminator.
    discriminator_fn: A discriminator function that conforms to TFGAN API.
    discriminator_scope: If not `None`, reuse discriminators from this scope.
    epsilon: A small positive number added for numerical stability when
      computing the gradient norm.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `real_data` and `generated_data`, and must be broadcastable to
      them (i.e., all dimensions must be either `1`, or the same as the
      corresponding dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.

  Returns:
    A loss Tensor. The shape depends on `reduction`.

  Raises:
    ValueError: If the rank of data Tensors is unknown.
  """
  real_data = ops.convert_to_tensor(real_data)
  generated_data = ops.convert_to_tensor(generated_data)
  if real_data.shape.ndims is None:
    raise ValueError('`real_data` can\'t have unknown rank.')
  if generated_data.shape.ndims is None:
    raise ValueError('`generated_data` can\'t have unknown rank.')

  differences = generated_data - real_data
  batch_size = differences.shape[0].value or array_ops.shape(differences)[0]
  alpha_shape = [batch_size] + [1] * (differences.shape.ndims - 1)
  alpha = random_ops.random_uniform(shape=alpha_shape)
  interpolates = real_data + (alpha * differences)

  # Reuse variables if a discriminator scope already exists.
  reuse = False if self.dis_scope.name is None else True
  with variable_scope.variable_scope(self.dis_scope.name, 'gpenalty_dscope', reuse=reuse):
    disc_interpolates = self.discriminator(interpolates)

  gradients = gradients_impl.gradients(disc_interpolates, interpolates)[0]
  gradient_squares = math_ops.reduce_sum(
      math_ops.square(gradients), axis=list(range(1, gradients.shape.ndims)))
  # Propagate shape information, if possible.
  if isinstance(batch_size, int):
    gradient_squares.set_shape([
        batch_size] + gradient_squares.shape.as_list()[1:])
  # For numerical stability, add epsilon to the sum before taking the square
  # root. Note tf.norm does not add epsilon.
  slopes = math_ops.sqrt(gradient_squares + epsilon)
  penalties = math_ops.square(slopes - 1.0)
  penalty = losses.compute_weighted_loss(penalties, weights, scope=scope)

  return penalty

def mutual_information_penalty(
    structured_generator_inputs,
    predicted_distributions,
    weights=1.0,
    scope=None,
    add_summaries=False):
  """Returns a penalty on the mutual information in an InfoGAN model.

  This loss comes from an InfoGAN paper https://arxiv.org/abs/1606.03657.

  Args:
    structured_generator_inputs: A list of Tensors representing the random noise
      that must  have high mutual information with the generator output. List
      length should match `predicted_distributions`.
    predicted_distributions: A list of tf.Distributions. Predicted by the
      recognizer, and used to evaluate the likelihood of the structured noise.
      List length should match `structured_generator_inputs`.
    weights: Optional `Tensor` whose rank is either 0, or the same dimensions as
      `structured_generator_inputs`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.

  Returns:
    A scalar Tensor representing the mutual information loss.
  """
  _validate_information_penalty_inputs(
      structured_generator_inputs, predicted_distributions)
  q_cat = predicted_distributions[0]
  q_cont = predicted_distributions[1]

  q_cat = ds.Categorical(q_cat)
  sigma_cont = tf.ones_like(q_cont)
  q_cont = ds.Normal(loc=q_cont, scale=sigma_cont)
  predicted_distributions = tf.concat([q_cat, q_cont], 1)
  # Calculate the negative log-likelihood of the reconstructed noise.
  log_probs = [math_ops.reduce_mean(dist.log_prob(noise)) for dist, noise in
               zip(predicted_distributions, structured_generator_inputs)]
  loss = -1 * losses.compute_weighted_loss(log_probs, weights, scope)

  return loss

def mean_square_loss(
    _input,
    _output,
    weights=1.0,
    scope=None,
    add_summaries=False):

  return 0.5 * tf.reduce_sum(tf.pow(tf.subtract(_input, _output), 2.0))



def _validate_distributions(distributions):
  if not isinstance(distributions, (list, tuple)):
    raise ValueError('`distributions` must be a list or tuple. Instead, '
                     'found %s.', type(distributions))
  for x in distributions:
    if not isinstance(x, ds.Distribution):
      raise ValueError('`distributions` must be a list of `Distributions`. '
                       'Instead, found %s.', type(x))


def _validate_information_penalty_inputs(
    structured_generator_inputs, predicted_distributions):
  """Validate input to `mutual_information_penalty`."""
  _validate_distributions(predicted_distributions)
  if len(structured_generator_inputs) != len(predicted_distributions):
    raise ValueError('`structured_generator_inputs` length %i must be the same '
                     'as `predicted_distributions` length %i.' % (
                         len(structured_generator_inputs),
                         len(predicted_distributions)))
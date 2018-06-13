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
    real_data,
    generated_data,
    generator_inputs,
    discriminator_fn,
    discriminator_scope,
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
  reuse = False if discriminator_scope is None else True
  with variable_scope.variable_scope(discriminator_scope, 'gpenalty_dscope',
                                     reuse=reuse):
    disc_interpolates = discriminator_fn(interpolates, generator_inputs)

  if isinstance(disc_interpolates, tuple):
    # ACGAN case: disc outputs more than one tensor
    disc_interpolates = disc_interpolates[0]

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
"""Helper functions for performing constrained optimization."""

import collections
import copy
import numpy as np
import tensorflow as tf


def project_multipliers_wrt_euclidean_norm(multipliers, radius):
    """Projects its argument onto the feasible region.
    The feasible region is the set of all vectors with nonnegative elements that
    sum to at most "radius".

    From https://github.com/google-research/tensorflow_constrained_optimization/blob/master/tensorflow_constrained_optimization/python/train/lagrangian_optimizer.py
    
    Args:
        multipliers: rank-1 `Tensor`, the Lagrange multipliers to project.
        radius: float, the radius of the feasible region.
    Returns:
        The rank-1 `Tensor` that results from projecting "multipliers" onto the
        feasible region w.r.t. the Euclidean norm.
    Raises:
        TypeError: if the "multipliers" `Tensor` is not floating-point.
        ValueError: if the "multipliers" `Tensor` does not have a fully-known shape,
          or is not one-dimensional.
    """
    if not multipliers.dtype.is_floating:
        raise TypeError("multipliers must have a floating-point dtype")
    multipliers_dims = multipliers.shape.dims
    if multipliers_dims is None:
        raise ValueError("multipliers must have a known rank")
    if len(multipliers_dims) != 1:
        raise ValueError("multipliers must be rank 1 (it is rank %d)" %
                     len(multipliers_dims))
    dimension = multipliers_dims[0].value
    if dimension is None:
        raise ValueError("multipliers must have a fully-known shape")

    def while_loop_condition(iteration, multipliers, inactive, old_inactive):
        """Returns false if the while loop should terminate."""
        del multipliers  # Needed by the body, but not the condition.
        not_done = (iteration < dimension)
        not_converged = tf.reduce_any(tf.not_equal(inactive, old_inactive))
        return tf.logical_and(not_done, not_converged)

    def while_loop_body(iteration, multipliers, inactive, old_inactive):
        """Performs one iteration of the projection."""
        del old_inactive  # Needed by the condition, but not the body.
        iteration += 1
        scale = tf.minimum(0.0, (radius - tf.reduce_sum(multipliers)) /
                           tf.maximum(1.0, tf.reduce_sum(inactive)))
        multipliers = multipliers + (scale * inactive)
        new_inactive = tf.cast(multipliers > 0, multipliers.dtype)
        multipliers = multipliers * new_inactive
        return (iteration, multipliers, new_inactive, inactive)

    iteration = tf.constant(0)
    inactive = tf.ones_like(multipliers, dtype=multipliers.dtype)

    # We actually want a do-while loop, so we explicitly call while_loop_body()
    # once before tf.while_loop().
    iteration, multipliers, inactive, old_inactive = while_loop_body(
      iteration, multipliers, inactive, inactive)
    iteration, multipliers, inactive, old_inactive = tf.while_loop(
      while_loop_condition,
      while_loop_body,
      loop_vars=(iteration, multipliers, inactive, old_inactive),
      name="euclidean_projection")

    return multipliers


def project_by_dykstra(weights, project_groups_fn, project_simplex_fn, num_iterations=1):
    """Applies dykstra's projection algorithm for monotonicity/trust constraints.

    Dykstra's alternating projections algorithm projects into intersection of
    several convex sets. For algorithm description itself use Google or Wiki:
    https://en.wikipedia.org/wiki/Dykstra%27s_projection_algorithm

    Returns honest projection with respect to L2 norm if num_iterations is inf.

    Args:
      weights: input vector representing flattend W matrix.
      project_groups_fn: function projecting W onto group linear equality constraints.
      project_simplex_fn: function projecting W onto probability simplex W1=1.
      num_iterations: number of iterations of Dykstra's algorithm.

    Returns:
      Projected weights tensor of same shape as `weights`.
    """
    if (num_iterations == 0):
        return weights
    
    def body(iteration, weights, last_change):
        """Body of the tf.while_loop for Dykstra's projection algorithm.

        This implements Dykstra's projection algorithm and requires rolling back
        the last projection change.

        Args:
          iteration: Iteration counter tensor.
          weights: Tensor with project weights at each iteraiton.
          last_change: Dict that stores the last change in the weights after
            projecting onto the each subset of constraints.

        Returns:
          The tuple (iteration, weights, last_change) at the end of each iteration.
        """
        last_change = copy.copy(last_change)
    
        # Project onto group linear equality constraints.
        rolled_back_weights = (weights - last_change["Aw=b"])
        weights = project_groups_fn(rolled_back_weights)
        last_change["Aw=b"] = weights - rolled_back_weights

        # Project onto simplex linear equality constraints.
        rolled_back_weights = (weights - last_change["1w=1"])
        weights = project_simplex_fn(rolled_back_weights)
        last_change["1w=1"] = weights - rolled_back_weights

        # Project onto nonnegativity constraints
        rolled_back_weights = weights - last_change["w>=0"]
        weights = tf.nn.relu(weights)
        last_change["w>=0"] = weights - rolled_back_weights

        return iteration + 1, weights, last_change

    def cond(iteration, weights, last_change):
        del weights, last_change
        return tf.less(iteration, num_iterations)

    # Run the body of the loop once to find required last_change keys. The set of
    # keys in the input and output of the body of tf.while_loop must be the same.
    # The resulting ops are discarded and will not be part of the TF graph.
    zeros = tf.zeros_like(weights)
    last_change = collections.defaultdict(lambda: zeros)
    (_, _, last_change) = body(0, weights, last_change)

    # Apply Dykstra's algorithm with tf.while_loop.
    iteration = tf.constant(0)
    last_change = {k: zeros for k in last_change}
    (_, weights, _) = tf.while_loop(cond, body, (iteration, weights, last_change))
    return weights


def project_multipliers_to_L1_ball(multipliers, center, radius):
    """Projects its argument onto the feasible region.
    The feasible region is the set of all vectors in the L1 ball with the given center multipliers and given radius.
    
    Args:
        multipliers: rank-1 `Tensor`, the Lagrange multipliers to project.
        radius: float, the radius of the feasible region.
        center: rank-1 `Tensor`, the Lagrange multipliers as the center.
    Returns:
        The rank-1 `Tensor` that results from projecting "multipliers" onto a L1 norm ball w.r.t. the Euclidean norm.
        The returned rank-1 `Tensor`  IS IN A SIMPLEX
    Raises:
        TypeError: if the "multipliers" `Tensor` is not floating-point.
        ValueError: if the "multipliers" `Tensor` does not have a fully-known shape,
          or is not one-dimensional.
    """
    assert radius >= 0
    # compute the offset from the center and the distance
    offset = tf.math.subtract(multipliers, center)
    dist = tf.math.abs(offset)
    # multipliers is not already a solution: optimum lies on the boundary (norm of dist == radius)
    # project *multipliers* on the simplex
    new_dist = project_multipliers_wrt_euclidean_norm(dist, radius=radius)
    signs = tf.math.sign(offset)
    new_offset =  tf.math.multiply(signs, new_dist)
    projection = tf.math.add(center, new_offset)
    projection = tf.maximum(0.0, projection)
    return projection
    
    
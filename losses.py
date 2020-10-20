"""Loss functions used for experiments."""

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


def concave_hinge_loss(labels, logits, weights=1.0, scope=None,
               loss_collection=ops.GraphKeys.LOSSES,
               reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Adds a concave hinge loss to the training procedure.

  This places a concave hinge relaxation over the original zero-one indicator loss:
  Indicator(labels*logits <= 0)

  Args:
    labels: The ground truth output tensor. Its shape should match the shape of
      logits. The values of the tensor are expected to be 0.0 or 1.0. Internally
      the {0,1} labels are converted to {-1,1} when calculating the hinge loss.
    logits: The logits, a float tensor. Note that logits are assumed to be
      unbounded and 0-centered. A value > 0 (resp. < 0) is considered a positive
      (resp. negative) binary prediction.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.
  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.
  Raises:
    ValueError: If the shapes of `logits` and `labels` don't match or
      if `labels` or `logits` is None.
  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  if labels is None:
    raise ValueError("labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with ops.name_scope(scope, "concave_hinge_loss", (logits, labels, weights)) as scope:
    logits = math_ops.cast(logits, dtype=dtypes.float32)
    labels = math_ops.cast(labels, dtype=dtypes.float32)
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    # We first need to convert binary labels to -1/1 labels (as floats).
    all_ones = array_ops.ones_like(labels)
    labels = math_ops.subtract(2 * labels, all_ones)
    losses = math_ops.subtract(all_ones, nn_ops.relu(
        math_ops.add(all_ones, math_ops.multiply(labels, logits))))
    return tf.compat.v1.losses.compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


def ramp_loss_lb(labels, logits, weights=1.0, scope=None,
               loss_collection=ops.GraphKeys.LOSSES,
               reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Adds a ramp loss to the training procedure.

  Lower bound version (interchangable with concave hinge).

  This places a relaxation over the original zero-one indicator loss:
  Indicator(labels*logits <= 0)

  Args:
    labels: The ground truth output tensor. Its shape should match the shape of
      logits. The values of the tensor are expected to be 0.0 or 1.0. Internally
      the {0,1} labels are converted to {-1,1} when calculating the hinge loss.
    logits: The logits, a float tensor. Note that logits are assumed to be
      unbounded and 0-centered. A value > 0 (resp. < 0) is considered a positive
      (resp. negative) binary prediction.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.
  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.
  Raises:
    ValueError: If the shapes of `logits` and `labels` don't match or
      if `labels` or `logits` is None.
  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  if labels is None:
    raise ValueError("labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with ops.name_scope(scope, "concave_hinge_loss", (logits, labels, weights)) as scope:
    logits = math_ops.cast(logits, dtype=dtypes.float32)
    labels = math_ops.cast(labels, dtype=dtypes.float32)
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    # We first need to convert binary labels to -1/1 labels (as floats).
    all_ones = array_ops.ones_like(labels)
    labels = math_ops.subtract(2 * labels, all_ones)
    losses_concave_hinge = math_ops.subtract(all_ones, nn_ops.relu(
        math_ops.add(all_ones, math_ops.multiply(labels, logits))))
    losses_ramp_lb = nn_ops.relu(losses_concave_hinge)
    return tf.compat.v1.losses.compute_weighted_loss(
        losses_ramp_lb, weights, scope, loss_collection, reduction=reduction)


def ramp_loss(labels, logits, weights=1.0, scope=None,
               loss_collection=ops.GraphKeys.LOSSES,
               reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Adds a ramp loss to the training procedure.

  This places a relaxation over the original zero-one indicator loss:
  Indicator(labels*logits <= 0)

  Args:
    labels: The ground truth output tensor. Its shape should match the shape of
      logits. The values of the tensor are expected to be 0.0 or 1.0. Internally
      the {0,1} labels are converted to {-1,1} when calculating the hinge loss.
    logits: The logits, a float tensor. Note that logits are assumed to be
      unbounded and 0-centered. A value > 0 (resp. < 0) is considered a positive
      (resp. negative) binary prediction.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.
  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.
  Raises:
    ValueError: If the shapes of `logits` and `labels` don't match or
      if `labels` or `logits` is None.
  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  if labels is None:
    raise ValueError("labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with ops.name_scope(scope, "hinge_loss", (logits, labels, weights)) as scope:
    logits = math_ops.cast(logits, dtype=dtypes.float32)
    labels = math_ops.cast(labels, dtype=dtypes.float32)
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    # We first need to convert binary labels to -1/1 labels (as floats).
    all_ones = array_ops.ones_like(labels)
    labels = math_ops.subtract(2 * labels, all_ones)
    losses_hinge = nn_ops.relu(
        math_ops.subtract(all_ones, math_ops.multiply(labels, logits)))
    losses_ramp = math_ops.subtract(all_ones, nn_ops.relu(all_ones-losses_hinge))
    return tf.compat.v1.losses.compute_weighted_loss(
        losses_ramp, weights, scope, loss_collection, reduction=reduction)


def error_rate(predictions, labels):
  """ Error rate for predictions and binary labels.

  Args:
    predictions: numpy array of floats representing predictions. Predictions are treated 
      as positive classification if value is >0, and negative classification if value is <= 0.
    labels: numpy array of floats representing labels. labels are also treated as positive 
      classification if value is >0, and negative classification if value is <= 0.

  Returns: float, error rate of predictions classifications compared to label classifications.
  """
  signed_labels = (
    (labels > 0).astype(np.float32) - (labels <= 0).astype(np.float32))
  numerator = (np.multiply(signed_labels, predictions) <= 0).sum()
  denominator = predictions.shape[0]
  return float(numerator) / float(denominator)

def tpr(df, label_column):
    """
    Measure the true positive rate.
    """
    tp = sum((df['predictions'] >= 0.0) & (df[label_column] > 0.5))
    lp = sum(df[label_column] > 0.5)
    return float(tp) / float(lp)

def fpr(df, label_column):
  """Measure the false positive rate."""
  fp = sum((df['predictions'] >= 0.0) & (df[label_column] < 0.5))
  ln = sum(df[label_column] < 0.5)
  return float(fp) / float(ln)


def get_error_rate_and_constraints(df, protected_columns, proxy_columns, label_column, max_diff=0.02, constraint='tpr_and_fpr'):
    """Computes the error and fairness violations. Currently only computes tpr violations.
    
    Args:
      df: dataframe containing 'predictions' column and LABEL_COLUMN, PROTECTED_COLUMNS, and PROXY_COLUMNS.
        predictions column is not required to be thresholded.
      constraint: string referring to the constraint to measure.
      max_diff: float representing the slack on the constraint. If constraint is 'tpr_and_fpr', the same max_diff is applied to both
        tpr and fpr.

    Returns: 
      true_G_constraints: if constraint is tpr_and_fpr, then this list has 2*m entries, where the first m entries are the tpr violations per group, 
        and the second m entries are the fpr violations per group.
    
    """
    error_rate_overall = error_rate(df[['predictions']], df[[label_column]])
    true_G_protected_dfs = [df[df[protected_attribute] > 0.5] for protected_attribute in protected_columns]
    proxy_Ghat_protected_dfs = [df[df[protected_attribute] > 0.5] for protected_attribute in proxy_columns]

    if constraint == 'tpr':
      tpr_overall = tpr(df, label_column)
      true_G_constraints = [tpr_overall - tpr(protected_df, label_column) - max_diff for protected_df in true_G_protected_dfs]
      proxy_Ghat_constraints = [tpr_overall - tpr(protected_df, label_column) - max_diff for protected_df in proxy_Ghat_protected_dfs]
    elif constraint == 'err':
      true_G_constraints = [error_rate(protected_df[['predictions']], protected_df[[label_column]]) - error_rate_overall - max_diff for protected_df in true_G_protected_dfs]
      proxy_Ghat_constraints = [error_rate(protected_df[['predictions']], protected_df[[label_column]]) - error_rate_overall - max_diff for protected_df in proxy_Ghat_protected_dfs]
    elif constraint == 'fpr':
      fpr_overall = fpr(df, label_column)
      true_G_constraints = [fpr(protected_df, label_column) - fpr_overall - max_diff for protected_df in true_G_protected_dfs]
      proxy_Ghat_constraints = [fpr(protected_df, label_column) - fpr_overall - max_diff for protected_df in proxy_Ghat_protected_dfs] 
    elif constraint == 'tpr_and_fpr':
      fpr_overall = fpr(df, label_column)
      tpr_overall = tpr(df, label_column)
      if type(max_diff) is list:
        max_diff_tpr = max_diff[0]
        max_diff_fpr = max_diff[1]
      else:
        max_diff_tpr = max_diff
        max_diff_fpr = max_diff
      true_G_constraints_tpr = [tpr_overall - tpr(protected_df, label_column) - max_diff_tpr for protected_df in true_G_protected_dfs]
      true_G_constraints_fpr = [fpr(protected_df, label_column) - fpr_overall - max_diff_fpr for protected_df in true_G_protected_dfs]
      true_G_constraints = true_G_constraints_tpr + true_G_constraints_fpr
      proxy_Ghat_constraints_tpr = [tpr_overall - tpr(protected_df, label_column) - max_diff_tpr for protected_df in proxy_Ghat_protected_dfs]
      proxy_Ghat_constraints_fpr = [fpr(protected_df, label_column) - fpr_overall - max_diff_fpr for protected_df in proxy_Ghat_protected_dfs] 
      proxy_Ghat_constraints = proxy_Ghat_constraints_tpr + proxy_Ghat_constraints_fpr
    return error_rate_overall, true_G_constraints, proxy_Ghat_constraints


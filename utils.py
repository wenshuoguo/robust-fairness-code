"""Helper functions for working with tensors."""

import matplotlib.pyplot as plt
import numpy as np
from random import uniform 
import random
from scipy.stats import rankdata
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


# Flips entries in binary input tensor. input_tensor must have only entries of 0 or 1.
def flip_binary_tensor(input_tensor):
    return tf.subtract(tf.ones_like(input_tensor), input_tensor)


def binary_mask_tensor(length, ones_start, ones_length):
    """Creates a constant tensor with ones from ones_start to ones_end,
    and zeros everywhere else.
    
    Args:
      length: length of output tensor
      ones_start: start index for ones.
      ones_length: number of ones to assign from ones_start.
    """
    assert(ones_start + ones_length <= length)
    mask = np.zeros(length)
    mask[ones_start:ones_start + ones_length] = np.ones(ones_length)
    return tf.convert_to_tensor(mask, dtype=tf.float32)


def flip_binary_array(input_array):
  return np.subtract(np.ones_like(input_array), input_array)


def binary_mask_array(length, ones_start, ones_length):
    """Creates a constant tensor with ones from ones_start to ones_end,
    and zeros everywhere else.
    
    Args:
      length: length of output tensor
      ones_start: start index for ones.
      ones_length: number of ones to assign from ones_start.
    """
    assert(ones_start + ones_length <= length)
    mask = np.zeros(length)
    mask[ones_start:ones_start + ones_length] = np.ones(ones_length)
    return mask


def find_best_candidate_index(objective_vector,
                              constraints_matrix,
                              rank_objectives=True,
                              max_constraints=True):
  """Heuristically finds the best candidate solution to a constrained problem.
  This function deals with the constrained problem:
  > minimize f(w)
  > s.t. g_i(w) <= 0 for all i in {0,1,...,m-1}
  Here, f(w) is the "objective function", and g_i(w) is the ith (of m)
  "constraint function". Given a set of n "candidate solutions"
  {w_0,w_1,...,w_{n-1}}, this function finds the "best" solution according
  to the following heuristic:
    1. If max_constraints=True, the m constraints are collapsed down to one
       constraint, for which the constraint violation is the maximum constraint
       violation over the m original constraints. Otherwise, we continue with m
       constraints.
    2. Across all models, the ith constraint violations (i.e. max{0, g_i(0)})
       are ranked, as are the objectives (if rank_objectives=True).
    3. Each model is then associated its MAXIMUM rank across all m constraints
       (and the objective, if rank_objectives=True).
    4. The model with the minimal maximum rank is then identified. Ties are
       broken using the objective function value.
    5. The index of this "best" model is returned.
  The "objective_vector" parameter should be a numpy array with shape (n,), for
  which objective_vector[i] = f(w_i). Likewise, "constraints_matrix" should be a
  numpy array with shape (n,m), for which constraints_matrix[i,j] = g_j(w_i).
  For more specifics, please refer to:
  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization". ALT'19.
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)
  This function implements the heuristic used for hyperparameter search in the
  experiments of Section 5.2.
  Args:
    objective_vector: numpy array of shape (n,), where n is the number of
      "candidate solutions". Contains the objective function values.
    constraints_matrix: numpy array of shape (m,n), where n is the number of
      "candidate solutions", and m is the number of constraints. Contains the
      constraint violation magnitudes.
    rank_objectives: bool, whether the objective function values should be
      included in the initial ranking step. If True, both the objective and
      constraints will be ranked. If False, only the constraints will be ranked.
      In either case, the objective function values will be used for
      tiebreaking.
    max_constraints: bool, whether we should collapse the m constraints down to
      one by maximizing over them, before searching for the best index.
  Returns:
    The index (in {0,1,...,n-1}) of the "best" model according to the above
    heuristic.
  Raises:
    ValueError: if "objective_vector" and "constraints_matrix" have inconsistent
      shapes.
  """
  nn, mm = np.shape(constraints_matrix)
  if (nn,) != np.shape(objective_vector):
    raise ValueError(
        "objective_vector must have shape (n,), and constraints_matrix (n, m),"
        " where n is the number of candidates, and m is the number of "
        "constraints")

  # If max_constraints is True, then we collapse the mm constraints down to one,
  # where this "one" is the maximum constraint violation across all mm
  # constraints.
  if mm > 1 and max_constraints:
    constraints_matrix = np.amax(constraints_matrix, axis=1, keepdims=True)
    mm = 1

  if rank_objectives:
    maximum_ranks = rankdata(objective_vector, method="min")
  else:
    maximum_ranks = np.zeros(nn, dtype=np.int64)
  for ii in xrange(mm):
    # Take the maximum of the constraint functions with zero, since we want to
    # rank the magnitude of constraint *violations*. If the constraint is
    # satisfied, then we don't care how much it's satisfied by (as a result, we
    # we expect all models satisfying a constraint to be tied at rank 1).
    ranks = rankdata(np.maximum(0.0, constraints_matrix[:, ii]), method="min")
    maximum_ranks = np.maximum(maximum_ranks, ranks)

  best_index = None
  best_rank = float("Inf")
  best_objective = float("Inf")
  for ii in xrange(nn):
    if maximum_ranks[ii] < best_rank:
      best_index = ii
      best_rank = maximum_ranks[ii]
      best_objective = objective_vector[ii]
    elif (maximum_ranks[ii] == best_rank) and (objective_vector[ii] <=
                                               best_objective):
      best_index = ii
      best_objective = objective_vector[ii]

  return best_index


######## Util functions for unaveraged results dicts. #########

# Adds best_idx results to results_dict.
def add_results_dict_best_idx(results_dict, best_index):
    columns_to_add = ['train_01_objective_vector', 'train_01_true_G_constraints_matrix', 'train_01_proxy_Ghat_constraints_matrix', 
                     'val_01_objective_vector', 'val_01_true_G_constraints_matrix', 'val_01_proxy_Ghat_constraints_matrix', 
                     'test_01_objective_vector', 'test_01_true_G_constraints_matrix', 'test_01_proxy_Ghat_constraints_matrix']
    for column in columns_to_add:
        results_dict['best_' + column] = results_dict[column][best_index]
    return results_dict

def add_results_dict_best_idx_robust(results_dict, best_index):
    columns_to_add = ['train_01_objective_vector', 'train_01_true_G_constraints_matrix', 'train_01_proxy_Ghat_constraints_matrix', 
                     'val_01_objective_vector', 'val_01_true_G_constraints_matrix', 'val_01_proxy_Ghat_constraints_matrix', 
                     'test_01_objective_vector', 'test_01_true_G_constraints_matrix', 'test_01_proxy_Ghat_constraints_matrix',
                      'train_robust_constraints_matrix', 'val_robust_constraints_matrix',
                      'test_robust_constraints_matrix']
    for column in columns_to_add:
        results_dict['best_' + column] = results_dict[column][best_index]
    return results_dict


# Expects results dicts without averaging.
def print_results_last_iter(results_dict):
    print("last iterate 0/1 objectives: (train, val, test)")
    print("%.4f, %.4f ,%.4f " % (results_dict['train_01_objective_vector'][-1], 
                                 results_dict['val_01_objective_vector'][-1],
                                 results_dict['test_01_objective_vector'][-1],))
    
    print("last iterate 0/1 true G constraints: (train, val, test)")
    print("%.4f, %.4f ,%.4f" % (max(results_dict['train_01_true_G_constraints_matrix'][-1]), 
                                max(results_dict['val_01_true_G_constraints_matrix'][-1]),
                                max(results_dict['test_01_true_G_constraints_matrix'][-1])))
    
    print("last iterate 0/1 proxy Ghat constraints: (train, val, test)")
    print("%.4f, %.4f ,%.4f" % (max(results_dict['train_01_proxy_Ghat_constraints_matrix'][-1]), 
                                max(results_dict['val_01_proxy_Ghat_constraints_matrix'][-1]),
                                max(results_dict['test_01_proxy_Ghat_constraints_matrix'][-1])))


# Expects results dicts without averaging.
def print_results_best_iter(results_dict):
    print("best iterate 0/1 objectives: (train, val, test)")
    print("%.4f, %.4f ,%.4f " % (results_dict['best_train_01_objective_vector'], 
                                 results_dict['best_val_01_objective_vector'],
                                 results_dict['best_test_01_objective_vector'],))
    
    print("best iterate 0/1 true G constraints: (train, val, test)")
    print("%.4f, %.4f ,%.4f" % (max(results_dict['best_train_01_true_G_constraints_matrix']), 
                                max(results_dict['best_val_01_true_G_constraints_matrix']),
                                max(results_dict['best_test_01_true_G_constraints_matrix'])))
    
    print("best iterate 0/1 proxy Ghat constraints: (train, val, test)")
    print("%.4f, %.4f ,%.4f" % (max(results_dict['best_train_01_proxy_Ghat_constraints_matrix']), 
                                max(results_dict['best_val_01_proxy_Ghat_constraints_matrix']),
                                max(results_dict['best_test_01_proxy_Ghat_constraints_matrix'])))


# # Expects results dicts without averaging.
# def print_results_avg_iter(results_dict, num_avg_iters=10):
#     print("avg iterate 0/1 objectives: (train, val, test)")
#     print("%.4f, %.4f ,%.4f " % (np.mean(np.array(results_dict['train_01_objective_vector'][-num_avg_iters:])), 
#                                  np.mean(np.array(results_dict['val_01_objective_vector'][-num_avg_iters:])),
#                                  np.mean(np.array(results_dict['test_01_objective_vector'][-num_avg_iters:]))))
    
#     print("avg iterate 0/1 true G constraints: (train, val, test)")
#     print("%.4f, %.4f ,%.4f" % (np.max(np.mean(np.array(results_dict['train_01_true_G_constraints_matrix'][-num_avg_iters:]), 
#                                 np.max(np.mean(np.array(results_dict['val_01_true_G_constraints_matrix'][-num_avg_iters:]),
#                                 np.max(np.mean(np.array(results_dict['test_01_true_G_constraints_matrix'][-num_avg_iters:])))
    
#     print("avg iterate 0/1 proxy Ghat constraints: (train, val, test)")
#     print("%.4f, %.4f ,%.4f" % (max(results_dict['best_train_01_proxy_Ghat_constraints_matrix']), 
#                                 max(results_dict['best_val_01_proxy_Ghat_constraints_matrix']),
#                                 max(results_dict['best_test_01_proxy_Ghat_constraints_matrix'])))


# Expects results dicts without averaging.
def plot_optimization(results_dict):
    fig, axs = plt.subplots(4, figsize=(5,20))
    axs[0].plot(results_dict['train_hinge_objective_vector'])
    axs[0].set_title('train_hinge_objective_vector')
    axs[1].plot(results_dict['train_hinge_constraints_matrix'])
    axs[1].set_title('train_hinge_constraints_matrix')
    axs[2].plot(results_dict['train_01_proxy_Ghat_constraints_matrix'])
    axs[2].set_title('train_01_proxy_Ghat_constraints_matrix')
    axs[3].plot(results_dict['train_01_true_G_constraints_matrix'])
    axs[3].set_title('train_01_true_G_constraints_matrix')
    plt.show()
 

######### Util functions for averaged results_dicts ################

# Outputs a results_dict with mean and standard dev for each metric for list results_dicts.
def average_results_dict_fn(results_dicts):
    average_results_dict = {}
    for metric in results_dicts[0]:
        all_metric_arrays = []
        orig_shape = np.array(results_dicts[0][metric]).shape
        for results_dict in results_dicts: 
            all_metric_arrays.append(np.array(results_dict[metric]).flatten())
        all_metric_arrays = np.array(all_metric_arrays)
        mean_metric_flattened = np.mean(all_metric_arrays, axis=0)
        mean_metric = mean_metric_flattened.reshape(orig_shape)
        std_metric_flattened = np.std(all_metric_arrays, ddof=1, axis=0)
        std_metric = std_metric_flattened.reshape(orig_shape)
        average_results_dict[metric] = (mean_metric, std_metric)
    return average_results_dict


# Expects averaged results_dict with means and standard deviations.
def print_avg_results_last_iter(results_dict, iter_num=-1):
    def get_max_mean_std(mean_std_tuple):
        max_idx = np.argmax(mean_std_tuple[0][-1])
        max_mean = mean_std_tuple[0][-1][max_idx]
        max_std = mean_std_tuple[1][-1][max_idx]
        return max_mean, max_std

    print("last iterate 0/1 objectives: (train, val, test)")
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (results_dict['train_01_objective_vector'][0][iter_num], results_dict['train_01_objective_vector'][1][iter_num], 
                                                         results_dict['val_01_objective_vector'][0][iter_num], results_dict['val_01_objective_vector'][1][iter_num],
                                                         results_dict['test_01_objective_vector'][0][iter_num], results_dict['test_01_objective_vector'][1][iter_num]))
    
    print("last iterate 0/1 true G constraints: (train, val, test)")
    train_max_mean_std = get_max_mean_std(results_dict['train_01_true_G_constraints_matrix'])
    val_max_mean_std = get_max_mean_std(results_dict['val_01_true_G_constraints_matrix'])
    test_max_mean_std = get_max_mean_std(results_dict['test_01_true_G_constraints_matrix'])
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_std[0], train_max_mean_std[1], 
                                                     val_max_mean_std[0], val_max_mean_std[1],
                                                     test_max_mean_std[0], test_max_mean_std[1]))
    
    print("last iterate 0/1 proxy Ghat constraints: (train, val, test)")
    train_max_mean_std = get_max_mean_std(results_dict['train_01_proxy_Ghat_constraints_matrix'])
    val_max_mean_std = get_max_mean_std(results_dict['val_01_proxy_Ghat_constraints_matrix'])
    test_max_mean_std = get_max_mean_std(results_dict['test_01_proxy_Ghat_constraints_matrix'])
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_std[0], train_max_mean_std[1], 
                                                     val_max_mean_std[0], val_max_mean_std[1],
                                                     test_max_mean_std[0], test_max_mean_std[1]))


# Expects averaged results_dict with means and standard deviations.
def print_avg_results_best_iter(results_dict):
    def get_max_mean_std_best(mean_std_tuple):
        max_idx = np.argmax(mean_std_tuple[0])
        max_mean = mean_std_tuple[0][max_idx]
        max_std = mean_std_tuple[1][max_idx]
        return max_mean, max_std

    print("best iterate 0/1 objectives: (train, val, test)")
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (float(results_dict['best_train_01_objective_vector'][0]), float(results_dict['best_train_01_objective_vector'][1]), 
                                                     float(results_dict['best_val_01_objective_vector'][0]), float(results_dict['best_val_01_objective_vector'][1]),
                                                     float(results_dict['best_test_01_objective_vector'][0]), float(results_dict['best_test_01_objective_vector'][1])))
    
    print("best iterate max 0/1 true G constraint violations: (train, val, test)")
    train_max_mean_std = get_max_mean_std_best(results_dict['best_train_01_true_G_constraints_matrix'])
    val_max_mean_std = get_max_mean_std_best(results_dict['best_val_01_true_G_constraints_matrix'])
    test_max_mean_std = get_max_mean_std_best(results_dict['best_test_01_true_G_constraints_matrix'])
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_std[0], train_max_mean_std[1], 
                                                         val_max_mean_std[0], val_max_mean_std[1],
                                                         test_max_mean_std[0], test_max_mean_std[1]))    

    print("best iterate max 0/1 proxy Ghat constraint violations: (train, val, test)")
    train_max_mean_std = get_max_mean_std_best(results_dict['best_train_01_proxy_Ghat_constraints_matrix'])
    val_max_mean_std = get_max_mean_std_best(results_dict['best_val_01_proxy_Ghat_constraints_matrix'])
    test_max_mean_std = get_max_mean_std_best(results_dict['best_test_01_proxy_Ghat_constraints_matrix'])
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_std[0], train_max_mean_std[1], 
                                                         val_max_mean_std[0], val_max_mean_std[1],
                                                         test_max_mean_std[0], test_max_mean_std[1]))


# Expects averaged results_dict with means and standard deviations.
def print_avg_results_best_iter_tpr_and_fpr(results_dict, num_groups=3):
    def get_max_mean_std_best(mean_std_tuple):
        means_tpr = mean_std_tuple[0][:num_groups]
        stds_tpr = mean_std_tuple[1][:num_groups]
        max_idx_tpr = np.argmax(means_tpr)
        max_mean_tpr = means_tpr[max_idx_tpr]
        max_std_tpr = stds_tpr[max_idx_tpr]

        means_fpr = mean_std_tuple[0][num_groups:]
        stds_fpr = mean_std_tuple[1][num_groups:]
        max_idx_fpr = np.argmax(means_fpr)
        max_mean_fpr = means_fpr[max_idx_fpr]
        max_std_fpr = stds_tpr[max_idx_fpr]
        return max_mean_tpr, max_std_tpr, max_mean_fpr, max_std_fpr

    print("best iterate 0/1 objectives: (train, val, test)")
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (float(results_dict['best_train_01_objective_vector'][0]), float(results_dict['best_train_01_objective_vector'][1]), 
                                                     float(results_dict['best_val_01_objective_vector'][0]), float(results_dict['best_val_01_objective_vector'][1]),
                                                     float(results_dict['best_test_01_objective_vector'][0]), float(results_dict['best_test_01_objective_vector'][1])))
    
    train_max_mean_tpr, train_max_std_tpr, train_max_mean_fpr, train_max_std_fpr = get_max_mean_std_best(results_dict['best_train_01_true_G_constraints_matrix'])
    val_max_mean_tpr, val_max_std_tpr, val_max_mean_fpr, val_max_std_fpr = get_max_mean_std_best(results_dict['best_val_01_true_G_constraints_matrix'])
    test_max_mean_tpr, test_max_std_tpr, test_max_mean_fpr, test_max_std_fpr = get_max_mean_std_best(results_dict['best_test_01_true_G_constraints_matrix'])

    print("best iterate max 0/1 true G constraint violations (TPR): (train, val, test)")
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_tpr, train_max_std_tpr, 
                                                         val_max_mean_tpr, val_max_std_tpr,
                                                         test_max_mean_tpr, test_max_std_tpr))   

    print("best iterate max 0/1 true G constraint violations (FPR): (train, val, test)")
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_fpr, train_max_std_fpr, 
                                                         val_max_mean_fpr, val_max_std_fpr,
                                                         test_max_mean_fpr, test_max_std_fpr))   

    train_max_mean_tpr, train_max_std_tpr, train_max_mean_fpr, train_max_std_fpr = get_max_mean_std_best(results_dict['best_train_01_proxy_Ghat_constraints_matrix'])
    val_max_mean_tpr, val_max_std_tpr, val_max_mean_fpr, val_max_std_fpr = get_max_mean_std_best(results_dict['best_val_01_proxy_Ghat_constraints_matrix'])
    test_max_mean_tpr, test_max_std_tpr, test_max_mean_fpr, test_max_std_fpr = get_max_mean_std_best(results_dict['best_test_01_proxy_Ghat_constraints_matrix'])

    print("best iterate max 0/1 proxy Ghat constraint violations (TPR): (train, val, test)")
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_tpr, train_max_std_tpr, 
                                                         val_max_mean_tpr, val_max_std_tpr,
                                                         test_max_mean_tpr, test_max_std_tpr))   

    print("best iterate max 0/1 proxy Ghat constraint violations (FPR): (train, val, test)")
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_fpr, train_max_std_fpr, 
                                                         val_max_mean_fpr, val_max_std_fpr,
                                                         test_max_mean_fpr, test_max_std_fpr))   
    
    if 'best_train_robust_constraints_matrix' in results_dict:
      train_max_mean_tpr, train_max_std_tpr, train_max_mean_fpr, train_max_std_fpr = get_max_mean_std_best(results_dict['best_train_robust_constraints_matrix'])
      val_max_mean_tpr, val_max_std_tpr, val_max_mean_fpr, val_max_std_fpr = get_max_mean_std_best(results_dict['best_val_robust_constraints_matrix'])
      test_max_mean_tpr, test_max_std_tpr, test_max_mean_fpr, test_max_std_fpr = get_max_mean_std_best(results_dict['best_test_robust_constraints_matrix'])
      
      print("best iterate max 0/1 robust constraint violations (TPR): (train, val, test)")
      print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_tpr, train_max_std_tpr, 
                                                           val_max_mean_tpr, val_max_std_tpr,
                                                           test_max_mean_tpr, test_max_std_tpr))   

      print("best iterate max 0/1 robust constraint violations (FPR): (train, val, test)")
      print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_fpr, train_max_std_fpr, 
                                                           val_max_mean_fpr, val_max_std_fpr,
                                                           test_max_mean_fpr, test_max_std_fpr))  


def print_avg_results_best_iter_robust(results_dict):
    def get_max_mean_std_best(mean_std_tuple):
        max_idx = np.argmax(mean_std_tuple[0])
        max_mean = mean_std_tuple[0][max_idx]
        max_std = mean_std_tuple[1][max_idx]
        return max_mean, max_std

    print("best iterate 0/1 objectives: (train, val, test)")
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (float(results_dict['best_train_01_objective_vector'][0]), float(results_dict['best_train_01_objective_vector'][1]), 
                                                     float(results_dict['best_val_01_objective_vector'][0]), float(results_dict['best_val_01_objective_vector'][1]),
                                                     float(results_dict['best_test_01_objective_vector'][0]), float(results_dict['best_test_01_objective_vector'][1])))
    
    print("best iterate max 0/1 true G constraint violations: (train, val, test)")
    train_max_mean_std = get_max_mean_std_best(results_dict['best_train_01_true_G_constraints_matrix'])
    val_max_mean_std = get_max_mean_std_best(results_dict['best_val_01_true_G_constraints_matrix'])
    test_max_mean_std = get_max_mean_std_best(results_dict['best_test_01_true_G_constraints_matrix'])
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_std[0], train_max_mean_std[1], 
                                                         val_max_mean_std[0], val_max_mean_std[1],
                             
                                                         test_max_mean_std[0], test_max_mean_std[1]))    
    print("best iterate max 0/1 proxy Ghat constraint violations: (train, val, test)")
    train_max_mean_std = get_max_mean_std_best(results_dict['best_train_01_proxy_Ghat_constraints_matrix'])
    val_max_mean_std = get_max_mean_std_best(results_dict['best_val_01_proxy_Ghat_constraints_matrix'])
    test_max_mean_std = get_max_mean_std_best(results_dict['best_test_01_proxy_Ghat_constraints_matrix'])
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_std[0], train_max_mean_std[1], 
                                                         val_max_mean_std[0], val_max_mean_std[1],
                                                         test_max_mean_std[0], test_max_mean_std[1]))

    print("best iterate max 0/1 robust constraint violations: (train, val, test)")
    train_max_mean_std = get_max_mean_std_best(results_dict['best_train_robust_constraints_matrix'])
    val_max_mean_std = get_max_mean_std_best(results_dict['best_val_robust_constraints_matrix'])
    test_max_mean_std = get_max_mean_std_best(results_dict['best_test_robust_constraints_matrix'])
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_std[0], train_max_mean_std[1], 
                                                         val_max_mean_std[0], val_max_mean_std[1],
                                                         test_max_mean_std[0], test_max_mean_std[1]))

# Expects averaged results_dict with means and standard deviations.
def print_avg_results_avg_iter(results_dict, num_avg_iters=10):
    def get_max_mean_std(mean_std_tuple):
        avg_means = np.mean(mean_std_tuple[0][-num_avg_iters:], axis=0)
        avg_stds = np.mean(mean_std_tuple[1][-num_avg_iters:], axis=0)
        max_idx = np.argmax(avg_means)
        max_avg_mean = avg_means[max_idx]
        max_avg_std = avg_stds[max_idx]
        return max_avg_mean, max_avg_std

    print("avg iterate 0/1 objectives: (train, val, test)")
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (np.mean(results_dict['train_01_objective_vector'][0][-num_avg_iters:]), np.mean(results_dict['train_01_objective_vector'][1][-num_avg_iters:]), 
                                                         np.mean(results_dict['val_01_objective_vector'][0][-num_avg_iters:]), np.mean(results_dict['val_01_objective_vector'][1][-num_avg_iters:]),
                                                         np.mean(results_dict['test_01_objective_vector'][0][-num_avg_iters:]), np.mean(results_dict['test_01_objective_vector'][1][-num_avg_iters:])))
    
    print("avg iterate 0/1 true G constraints: (train, val, test)")
    train_max_mean_std = get_max_mean_std(results_dict['train_01_true_G_constraints_matrix'])
    val_max_mean_std = get_max_mean_std(results_dict['val_01_true_G_constraints_matrix'])
    test_max_mean_std = get_max_mean_std(results_dict['test_01_true_G_constraints_matrix'])
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_std[0], train_max_mean_std[1], 
                                                     val_max_mean_std[0], val_max_mean_std[1],
                                                     test_max_mean_std[0], test_max_mean_std[1]))
    
    print("avg iterate 0/1 proxy Ghat constraints: (train, val, test)")
    train_max_mean_std = get_max_mean_std(results_dict['train_01_proxy_Ghat_constraints_matrix'])
    val_max_mean_std = get_max_mean_std(results_dict['val_01_proxy_Ghat_constraints_matrix'])
    test_max_mean_std = get_max_mean_std(results_dict['test_01_proxy_Ghat_constraints_matrix'])
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_std[0], train_max_mean_std[1], 
                                                     val_max_mean_std[0], val_max_mean_std[1],
                                                     test_max_mean_std[0], test_max_mean_std[1]))


# Expects averaged results_dict with means and standard deviations.
def plot_optimization_avg(results_dict, protected_columns, proxy_columns):
    fig, axs = plt.subplots(4, figsize=(5,20))
    num_iters = len(results_dict['train_hinge_objective_vector'][0])
    iters = np.arange(num_iters)
    axs[0].errorbar(iters, results_dict['train_hinge_objective_vector'][0], yerr=results_dict['train_hinge_objective_vector'][1])
    axs[0].set_title('train_hinge_objective_vector')
    for i in range(len(protected_columns)):
        axs[1].errorbar(iters, results_dict['train_hinge_constraints_matrix'][0].T[i], results_dict['train_hinge_constraints_matrix'][1].T[i], label=protected_columns[i])
        axs[2].errorbar(iters, results_dict['train_01_proxy_Ghat_constraints_matrix'][0].T[i], results_dict['train_01_proxy_Ghat_constraints_matrix'][1].T[i], label=proxy_columns[i])
        axs[3].errorbar(iters, results_dict['train_01_true_G_constraints_matrix'][0].T[i], results_dict['train_01_true_G_constraints_matrix'][1].T[i], label=protected_columns[i])
    axs[1].set_title('train_hinge_constraints_matrix')
    axs[1].legend()
    axs[2].set_title('train_01_proxy_Ghat_constraints_matrix')
    axs[2].legend()
    axs[3].set_title('train_01_true_G_constraints_matrix')
    axs[3].legend()
    plt.show()

def generate_rand_vec_l1_ball(center, radius):
    '''
    generate a random vector IN THE SIMPLEX, and in the l1 ball given the center and radius
    '''
    n = len(center)
    splits = [0] + [uniform(0, 1) for _ in range(0,n-1)] + [1]
    splits.sort()
    diffs = [x - splits[i - 1] for i, x in enumerate(splits)][1:]
    diffs = map(lambda x:x*radius, diffs)
    diffs = np.array(list(diffs))
    signs = [(-1)**random.randint(0,1) for i in range(n)]
    diffs = np.multiply(diffs, signs)
    result = np.add(diffs, center)
    result = np.maximum(result, np.zeros(n))
    return result
    

def print_noise_priors(df, protected_columns, proxy_columns):
    """Prints noise priors including P(Ghat != G | G = j) for all j."""
    total_numerator = 0
    for i in range(len(protected_columns)):
        print("P(Ghat != G | G = j) where j = ", protected_columns[i])
        numerator = np.sum(np.multiply(df[protected_columns[i]],df[proxy_columns[i]]))
        total_numerator += numerator
        denominator = np.sum(df[protected_columns[i]])
        print(1 - numerator/denominator)
    print("P(Ghat != G)")
    print(1 - total_numerator/len(df))
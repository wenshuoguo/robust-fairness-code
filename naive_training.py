""" Functions for training using the naive approach."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import time

import data
import losses
import optimization
import model
import utils

class NaiveModel(model.Model):
    """Linear model for performing constrained optimization with naive approach.
    
    Args:
      feature_names: list of strings, a list of names of all feature columns.
      protected_columns: list of strings, a list of the names of all protected group columns 
        (column should contain values of 0 or 1 representing group membership).
      label_column: string, name of label column. Column should contain values of 0 or 1.
      maximum_lambda_radius: float, an optional upper bound to impose on the
        sum of the lambdas.
    
    Raises:
      ValueError: if "maximum_lambda_radius" is nonpositive.  
    """   
    def build_train_ops(self, constraint = 'tpr', learning_rate_theta=0.01, learning_rate_lambda=0.01, constraints_slack=1.0):
        """Builds operators that take gradient steps during training.
        
        Args: 
          learning_rate_theta: float, learning rate for theta parameter on descent step.
          learning_rate_lambda: float, learning rate for lambda parameter on ascent step.
          constraints_slack: float, amount of slack for constraints. New constraint will be
              original_constraint - constraints_slack
        
        """
        # Hinge loss objective.
        self.objective = tf.losses.hinge_loss(self.labels_placeholder, self.predictions_tensor)
        constraints_list = []
        if constraint == 'fpr':
            constraints_list = self.get_equal_fpr_constraints(constraints_slack=constraints_slack)
        elif constraint == 'err':
            constraints_list = self.get_equal_accuracy_constraints(constraints_slack=constraints_slack)
        elif constraint == 'fpr_nonconv':
            constraints_list = self.get_equal_fpr_constraints_nonconvex(constraints_slack=constraints_slack)
        elif constraint == 'tpr':
            constraints_list = self.get_equal_tpr_constraints(constraints_slack=constraints_slack)
        elif constraint == 'tpr_nonconv':
            constraints_list = self.get_equal_tpr_constraints_nonconvex(constraints_slack=constraints_slack)
        elif constraint == 'tpr_and_fpr':
            constraints_list = self.get_equal_tpr_and_fpr_constraints(constraints_slack=constraints_slack)
        else: 
            raise("constraint %s not supported." % (constraint))
        self.num_constraints = len(constraints_list)
        self.constraints = tf.convert_to_tensor(constraints_list)
        
        # Create lagrange multiplier variables.
        initial_lambdas = np.zeros((self.num_constraints,), dtype=np.float32)
        self.lambda_variables = tf.compat.v2.Variable(
          initial_lambdas,
          trainable=True,
          name="lambdas",
          dtype=tf.float32, 
          constraint=self.project_lambdas)
        
        lagrangian_loss = self.objective + tf.tensordot(
          tf.cast(self.lambda_variables, dtype=self.constraints.dtype.base_dtype),
          self.constraints, 1)

        optimizer_theta = tf.train.AdamOptimizer(learning_rate_theta)
        optimizer_lambda = tf.train.AdamOptimizer(learning_rate_lambda)

        self.train_op_theta = optimizer_theta.minimize(lagrangian_loss, var_list=self.theta_variables)
        self.train_op_lambda = optimizer_lambda.minimize(-lagrangian_loss, var_list=self.lambda_variables)
        return self.train_op_theta, self.train_op_lambda


def training_generator(model,
                       train_df,
                       val_df,
                       test_df,
                       minibatch_size=None,
                       num_iterations_per_loop=1,
                       num_loops=1):
    tf.set_random_seed(31337)
    num_rows = train_df.shape[0]
    if minibatch_size is None:
        minibatch_size = num_rows
    else:
        minibatch_size = min(minibatch_size, num_rows)
    permutation = list(range(train_df.shape[0]))
    random.shuffle(permutation)

    session = tf.Session()
    session.run((tf.global_variables_initializer(),
               tf.local_variables_initializer()))

    # Iterate through minibatches. Gradients are computed on each minibatch.
    minibatch_start_index = 0
    for n in range(num_loops):
        for _ in range(num_iterations_per_loop):
            minibatch_indices = []
            while len(minibatch_indices) < minibatch_size:
                minibatch_end_index = (
                minibatch_start_index + minibatch_size - len(minibatch_indices))
                if minibatch_end_index >= num_rows:
                    minibatch_indices += range(minibatch_start_index, num_rows)
                    minibatch_start_index = 0
                else:
                    minibatch_indices += range(minibatch_start_index, minibatch_end_index)
                    minibatch_start_index = minibatch_end_index
            minibatch_df = train_df.iloc[[permutation[ii] for ii in minibatch_indices]]
            # Descent step on theta.
            session.run(
                  model.train_op_theta,
                  feed_dict=model.feed_dict_helper(minibatch_df))
            # Ascent step on lambda.
            session.run(
                  model.train_op_lambda,
                  feed_dict=model.feed_dict_helper(minibatch_df))

        objective = session.run(model.objective, model.feed_dict_helper(train_df))
        constraints = session.run(model.constraints, model.feed_dict_helper(train_df))
        train_predictions = session.run(
            model.predictions_tensor,
            feed_dict=model.feed_dict_helper(train_df))
        val_predictions = session.run(
            model.predictions_tensor,
            feed_dict=model.feed_dict_helper(val_df))
        test_predictions = session.run(
            model.predictions_tensor,
            feed_dict=model.feed_dict_helper(test_df))
        lambda_variables = session.run(model.lambda_variables)

        yield (objective, constraints, train_predictions, lambda_variables, val_predictions, test_predictions)


def training_helper(model,
                    train_df,
                    val_df,
                    test_df,
                    protected_columns, 
                    proxy_columns, 
                    label_column,
                    minibatch_size = None,
                    num_iterations_per_loop=1,
                    num_loops=1,
                    constraint='tpr',
                    max_diff=0.03):
    train_hinge_objective_vector = []
    # Hinge loss constraint violations on the proxy groups.
    train_hinge_constraints_matrix = []
    train_01_objective_vector = []
    train_01_true_G_constraints_matrix = []
    train_01_proxy_Ghat_constraints_matrix = []
    lambda_variables_matrix = []
    
    val_01_objective_vector = []
    val_01_true_G_constraints_matrix = []
    val_01_proxy_Ghat_constraints_matrix = []
    
    # List of T scalar values representing the 01 objective at each iteration.
    test_01_objective_vector = []
    # List of T vectors of size m, where each vector[i] is the zero-one constraint violation for group i.
    # Eventually we will just pick the last vector in this list, and take the max over m entries to get the max constraint violation.
    test_01_true_G_constraints_matrix = []
    test_01_proxy_Ghat_constraints_matrix = []
    for objective, constraints, train_predictions, lambda_variables, val_predictions, test_predictions in training_generator(
      model, train_df, val_df, test_df, minibatch_size, num_iterations_per_loop,
      num_loops):
        train_hinge_objective_vector.append(objective)
        train_hinge_constraints_matrix.append(constraints)
        
        train_df['predictions'] = train_predictions
        train_01_objective, train_01_true_G_constraints, train_01_proxy_Ghat_constraints = losses.get_error_rate_and_constraints(train_df, protected_columns, proxy_columns, label_column, constraint=constraint, max_diff=max_diff)
        train_01_objective_vector.append(train_01_objective)
        train_01_true_G_constraints_matrix.append(train_01_true_G_constraints)
        train_01_proxy_Ghat_constraints_matrix.append(train_01_proxy_Ghat_constraints)
        
        lambda_variables_matrix.append(lambda_variables)
        
        val_df['predictions'] = val_predictions
        val_01_objective, val_01_true_G_constraints, val_01_proxy_Ghat_constraints = losses.get_error_rate_and_constraints(val_df, protected_columns, proxy_columns, label_column, constraint=constraint, max_diff=max_diff) 
        val_01_objective_vector.append(val_01_objective)
        val_01_true_G_constraints_matrix.append(val_01_true_G_constraints)
        val_01_proxy_Ghat_constraints_matrix.append(val_01_proxy_Ghat_constraints)
        
        test_df['predictions'] = test_predictions
        test_01_objective, test_01_true_G_constraints, test_01_proxy_Ghat_constraints = losses.get_error_rate_and_constraints(test_df, protected_columns, proxy_columns, label_column, constraint=constraint, max_diff=max_diff) 
        test_01_objective_vector.append(test_01_objective)
        test_01_true_G_constraints_matrix.append(test_01_true_G_constraints)
        test_01_proxy_Ghat_constraints_matrix.append(test_01_proxy_Ghat_constraints)
        
    return {'train_hinge_objective_vector': train_hinge_objective_vector, 
            'train_hinge_constraints_matrix': train_hinge_constraints_matrix, 
            'train_01_objective_vector': train_01_objective_vector, 
            'train_01_true_G_constraints_matrix': train_01_true_G_constraints_matrix, 
            'train_01_proxy_Ghat_constraints_matrix': train_01_proxy_Ghat_constraints_matrix, 
            'lambda_variables_matrix': lambda_variables_matrix, 
            'val_01_objective_vector': val_01_objective_vector, 
            'val_01_true_G_constraints_matrix': val_01_true_G_constraints_matrix, 
            'val_01_proxy_Ghat_constraints_matrix': val_01_proxy_Ghat_constraints_matrix,
            'test_01_objective_vector': test_01_objective_vector, 
            'test_01_true_G_constraints_matrix': test_01_true_G_constraints_matrix, 
            'test_01_proxy_Ghat_constraints_matrix': test_01_proxy_Ghat_constraints_matrix}


def get_results_for_learning_rates(input_df, 
                                    feature_names, protected_columns, proxy_columns, label_column, 
                                    constraint = 'tpr', 
                                    learning_rates_theta = [0.001,0.01,0.1], 
                                    learning_rates_lambda = [0.5, 1, 2], 
                                    num_runs=10, 
                                    minibatch_size=None, 
                                    num_iterations_per_loop=10, 
                                    num_loops=500, 
                                    constraints_slack=1.0,
                                    max_diff=0.05,
                                    hidden_layer_size=0):    
    ts = time.time()
    # 10 runs with mean and stddev
    results_dicts_runs = []
    for i in range(num_runs):
        print('Split %d of %d' % (i, num_runs))
        t_split = time.time()
        train_df, val_df, test_df = data.train_val_test_split(input_df, 0.6, 0.2, seed=10+i)
        val_objectives = []
        val_constraints_matrix = []
        results_dicts = []
        learning_rates_iters_theta = []
        learning_rates_iters_lambda = []
        for learning_rate_theta in learning_rates_theta:
            for learning_rate_lambda in learning_rates_lambda:
                t_start_iter = time.time() - ts
                print("time since start:", t_start_iter)
                print("optimizing learning rate theta: %.3f learning rate lambda: %.3f" % (learning_rate_theta, learning_rate_lambda))
                model = NaiveModel(feature_names, proxy_columns, label_column, maximum_lambda_radius=1,hidden_layer_size=hidden_layer_size)
                model.build_train_ops(constraint=constraint, learning_rate_theta=learning_rate_theta, learning_rate_lambda=learning_rate_lambda, constraints_slack=constraints_slack)

                # training_helper returns the list of errors and violations over each epoch.
                results_dict = training_helper(
                      model,
                      train_df,
                      val_df,
                      test_df,
                      protected_columns, 
                      proxy_columns, 
                      label_column,
                      minibatch_size=minibatch_size,
                      num_iterations_per_loop=num_iterations_per_loop,
                      num_loops=num_loops,
                      constraint=constraint,
                      max_diff=max_diff)
                
                best_index_iters = utils.find_best_candidate_index(np.array(results_dict['train_01_objective_vector']),np.array(results_dict['train_01_proxy_Ghat_constraints_matrix']), rank_objectives=True, max_constraints=True)
                val_objectives.append(results_dict['val_01_objective_vector'][best_index_iters])
                val_constraints_matrix.append(results_dict['val_01_proxy_Ghat_constraints_matrix'][best_index_iters])
                print ("best val objective: %0.4f" % results_dict['val_01_objective_vector'][best_index_iters])
                print ("best val constraints:", results_dict['val_01_proxy_Ghat_constraints_matrix'][best_index_iters])
                results_dict_best_idx = utils.add_results_dict_best_idx(results_dict, best_index_iters)
                results_dicts.append(results_dict_best_idx)
                learning_rates_iters_theta.append(learning_rate_theta)
                learning_rates_iters_lambda.append(learning_rate_lambda)
                print("Finished optimizing learning rate theta: %.3f learning rate lambda: %.3f" % (learning_rate_theta, learning_rate_lambda))
                print("Time that this run took:", time.time() - t_start_iter - ts)
                
        best_index = utils.find_best_candidate_index(np.array(val_objectives),np.array(val_constraints_matrix), rank_objectives=True, max_constraints=True)
        best_results_dict = results_dicts[best_index]
        best_learning_rate_theta = learning_rates_iters_theta[best_index]
        best_learning_rate_lambda = learning_rates_iters_lambda[best_index]
        print('best_learning_rate_theta,', best_learning_rate_theta)
        print('best_learning_rate_lambda', best_learning_rate_lambda)
        print('best true G constraint violations', best_results_dict['best_train_01_true_G_constraints_matrix'])
        results_dicts_runs.append(best_results_dict)
        print("time it took for this split", time.time() - t_split)
        
    final_average_results_dict = utils.average_results_dict_fn(results_dicts_runs)
    
    return final_average_results_dict


def train_one_model(input_df, 
                    feature_names, protected_columns, proxy_columns, label_column, 
                    constraint = 'tpr', 
                    learning_rate_theta = 0.01, 
                    learning_rate_lambda = 1, 
                    minibatch_size=1000, 
                    num_iterations_per_loop=100, 
                    num_loops=30, 
                    constraints_slack=1.3):    
    train_df, val_df, test_df = data.train_val_test_split(input_df, 0.6, 0.2, seed=88)
    model = NaiveModel(feature_names, proxy_columns, label_column, maximum_lambda_radius=2)
    model.build_train_ops(constraint=constraint, learning_rate_theta=learning_rate_theta, learning_rate_lambda=learning_rate_lambda, constraints_slack=constraints_slack)

    # training_helper returns the list of errors and violations over each epoch.
    results_dict = training_helper(
          model,
          train_df,
          val_df,
          test_df,
          protected_columns, 
          proxy_columns, 
          label_column,
          constraint=constraint,
          minibatch_size=minibatch_size,
          num_iterations_per_loop=num_iterations_per_loop,
          num_loops=num_loops)

    best_index_iters = utils.find_best_candidate_index(np.array(results_dict['train_01_objective_vector']),np.array(results_dict['train_01_proxy_Ghat_constraints_matrix']))
    results_dict_best_idx = utils.add_results_dict_best_idx(results_dict, best_index_iters)
    return results_dict_best_idx, (train_df, val_df, test_df)


# Expects averaged results_dict with means and standard deviations.
def print_avg_results_last_iter(results_dict):
    def get_max_mean_std(mean_std_tuple):
        max_idx = np.argmax(mean_std_tuple[0][-1])
        max_mean = mean_std_tuple[0][-1][max_idx]
        max_std = mean_std_tuple[1][-1][max_idx]
        return max_mean, max_std

    print("last iterate 0/1 objectives: (train, val, test)")
    print("%.4f $\pm$ %.4f,%.4f $\pm$ %.4f,%.4f $\pm$ %.4f" % (results_dict['train_01_objective_vector'][0][-1], results_dict['train_01_objective_vector'][1][-1], 
                                                         results_dict['val_01_objective_vector'][0][-1], results_dict['val_01_objective_vector'][1][-1],
                                                         results_dict['test_01_objective_vector'][0][-1], results_dict['test_01_objective_vector'][1][-1]))
    
    print("last iterate 0/1 true G constraints: (train, val, test)")
    train_max_mean_std = get_max_mean_std(results_dict['train_01_true_G_constraints_matrix'])
    val_max_mean_std = get_max_mean_std(results_dict['val_01_true_G_constraints_matrix'])
    test_max_mean_std = get_max_mean_std(results_dict['test_01_true_G_constraints_matrix'])
    print("%.4f $\pm$ %.4f,%.4f $\pm$ %.4f,%.4f $\pm$ %.4f" % (train_max_mean_std[0], train_max_mean_std[1], 
                                                     val_max_mean_std[0], val_max_mean_std[1],
                                                     test_max_mean_std[0], test_max_mean_std[1]))
    
    print("last iterate 0/1 proxy Ghat constraints: (train, val, test)")
    train_max_mean_std = get_max_mean_std(results_dict['train_01_proxy_Ghat_constraints_matrix'])
    val_max_mean_std = get_max_mean_std(results_dict['val_01_proxy_Ghat_constraints_matrix'])
    test_max_mean_std = get_max_mean_std(results_dict['test_01_proxy_Ghat_constraints_matrix'])
    print("%.4f $\pm$ %.4f,%.4f $\pm$ %.4f,%.4f $\pm$ %.4f" % (train_max_mean_std[0], train_max_mean_std[1], 
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
    print("%.4f $\pm$ %.4f,%.4f $\pm$ %.4f,%.4f $\pm$ %.4f" % (float(results_dict['best_train_01_objective_vector'][0]), float(results_dict['best_train_01_objective_vector'][1]), 
                                                     float(results_dict['best_val_01_objective_vector'][0]), float(results_dict['best_val_01_objective_vector'][1]),
                                                     float(results_dict['best_test_01_objective_vector'][0]), float(results_dict['best_test_01_objective_vector'][1])))
    
    print("best iterate max 0/1 true G constraint violations: (train, val, test)")
    train_max_mean_std = get_max_mean_std_best(results_dict['best_train_01_true_G_constraints_matrix'])
    val_max_mean_std = get_max_mean_std_best(results_dict['best_val_01_true_G_constraints_matrix'])
    test_max_mean_std = get_max_mean_std_best(results_dict['best_test_01_true_G_constraints_matrix'])
    print("%.4f $\pm$ %.4f,%.4f $\pm$ %.4f,%.4f $\pm$ %.4f" % (train_max_mean_std[0], train_max_mean_std[1], 
                                                         val_max_mean_std[0], val_max_mean_std[1],
                                                         test_max_mean_std[0], test_max_mean_std[1]))    

    print("best iterate max 0/1 proxy Ghat constraint violations: (train, val, test)")
    train_max_mean_std = get_max_mean_std_best(results_dict['best_train_01_proxy_Ghat_constraints_matrix'])
    val_max_mean_std = get_max_mean_std_best(results_dict['best_val_01_proxy_Ghat_constraints_matrix'])
    test_max_mean_std = get_max_mean_std_best(results_dict['best_test_01_proxy_Ghat_constraints_matrix'])
    print("%.4f $\pm$ %.4f,%.4f $\pm$ %.4f,%.4f $\pm$ %.4f" % (train_max_mean_std[0], train_max_mean_std[1], 
                                                         val_max_mean_std[0], val_max_mean_std[1],
                                                         test_max_mean_std[0], test_max_mean_std[1]))


# Expects results dicts without averaging.
def print_results_avg_iter(results_dict, num_avg_iters=10):
    print("avg iterate 0/1 objectives: (train, val, test)")
    print("%.4f, %.4f ,%.4f " % (np.mean(np.array(results_dict['train_01_objective_vector'][-num_avg_iters:])), 
                                 np.mean(np.array(results_dict['val_01_objective_vector'][-num_avg_iters:])),
                                 np.mean(np.array(results_dict['test_01_objective_vector'][-num_avg_iters:]))))
    
    print("avg iterate 0/1 true G constraints: (train, val, test)")
    print("%.4f, %.4f ,%.4f" % (np.max(np.mean(np.array(results_dict['train_01_true_G_constraints_matrix'][-num_avg_iters:]), axis=0)), 
                                np.max(np.mean(np.array(results_dict['val_01_true_G_constraints_matrix'][-num_avg_iters:]), axis=0)),
                                np.max(np.mean(np.array(results_dict['test_01_true_G_constraints_matrix'][-num_avg_iters:]), axis=0))))
    
    print("avg iterate 0/1 proxy Ghat constraints: (train, val, test)")
    print("%.4f, %.4f ,%.4f" % (np.max(np.mean(np.array(results_dict['train_01_proxy_Ghat_constraints_matrix'][-num_avg_iters:]), axis=0)), 
                                np.max(np.mean(np.array(results_dict['val_01_proxy_Ghat_constraints_matrix'][-num_avg_iters:]), axis=0)),
                                np.max(np.mean(np.array(results_dict['test_01_proxy_Ghat_constraints_matrix'][-num_avg_iters:]), axis=0))))

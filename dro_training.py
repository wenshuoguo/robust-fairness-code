""" Functions for training using the naive approach."""

import os

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import time

import data
import losses
import model
import naive_training
import optimization
import utils


class DRO_Model(model.Model):
    """Linear model with DRO constrained optimization.
    
     Args:
      feature_names: list of strings, a list of names of all feature columns.
      protected_columns: list of strings, a list of the names of all protected group columns 
        (column should contain values of 0 or 1 representing group membership).
      label_column: string, name of label column. Column should contain values of 0 or 1.
      maximum_lambda_radius: float, an optional upper bound to impose on the
        sum of the lambdas.
      maximum_p_radius: float, an optional upper bound to impose on the
        L1 norm of each row of phats - ptildes.
    
    Raises:
      ValueError: if "maximum_lambda_radius" is nonpositive.  
      ValueError: if "maximum_p_radius" is negative.  
    """
    def __init__(self, feature_names, protected_columns, label_column, phats, maximum_lambda_radius=None, maximum_p_radius=[1,1,1]):
        tf.reset_default_graph()
        tf.random.set_random_seed(123)
        
        self.feature_names = feature_names
        self.protected_columns = protected_columns
        self.label_column = label_column
        self.num_data = len(phats[0])
        self.phats = phats
        
        if (maximum_lambda_radius is not None and maximum_lambda_radius <= 0.0):
            raise ValueError("maximum_lambda_radius must be strictly positive")
        if (maximum_p_radius is not None and maximum_p_radius[0] < 0.0):
            raise ValueError("maximum_p_radius must be non negative")
        self._maximum_lambda_radius = maximum_lambda_radius
        self._maximum_p_radius = maximum_p_radius
        
        # Set up feature and label tensors.
        num_features = len(self.feature_names)
        self.features_placeholder = tf.placeholder(tf.float32, shape=(None, num_features), name='features_placeholder')
        self.protected_placeholders = [tf.placeholder(tf.float32, shape=(None, 1), name=attribute+"_placeholder") for attribute in self.protected_columns]
        self.labels_placeholder = tf.placeholder(
            tf.float32, shape=(None, 1), name='labels_placeholder')
        self.num_groups = len(self.protected_placeholders)
        
        # Construct linear model.
        self.predictions_tensor = tf.layers.dense(inputs=self.features_placeholder, units=1, activation=None, name="linear_model")
        self.theta_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "linear_model")
    
    def build_train_ops(self, constraint= 'tpr', learning_rate_theta=0.01, learning_rate_lambda=0.01, learning_rate_p_list=[0.01, 0.01, 0.01], constraints_slack=1.0):
        """Builds operators that take gradient steps during training.
        
        Args: 
          learning_rate_theta: float, learning rate for theta parameter on descent step.
          learning_rate_lambda: float, learning rate for lambda parameter on ascent step.
          learning_rate_p_list: list of float, learning rate for ptilde parameters on ascent step.
          constraints_slack: float, amount of slack for constraints. New constraint will be
              original_constraint - constraints_slack
        
        """
        # Hinge loss objective.
        self.objective = tf.losses.hinge_loss(self.labels_placeholder, self.predictions_tensor)
        
        # Create lagrange multiplier variables lambda.
        initial_lambdas = np.zeros((self.num_groups,), dtype=np.float32)
        if constraint == 'tpr_and_fpr':
            initial_lambdas = np.zeros((2*self.num_groups,), dtype=np.float32)
        self.lambda_variables = tf.compat.v2.Variable(
          initial_lambdas,
          trainable=True,
          name="lambdas",
          dtype=tf.float32, 
          constraint=self.project_lambdas)
        # Create lagrange multiplier variables p.
        self.p_variables_list = []
        
        def make_projection_p(i2):
            return lambda x: self.project_ptilde(x, i2)
        
        for i in range(self.num_groups):
            initial_p = np.zeros((self.num_data,), dtype=np.float32)
            self.p_variables_list.append(tf.compat.v2.Variable(
              initial_p,
              trainable=True,
              name="ptilde",
              dtype=tf.float32, 
              constraint=make_projection_p(i)))
        
        constraints_list = []
        if constraint == 'tpr':
            constraints_list = self.get_equal_tpr_constraints_dro(constraints_slack=constraints_slack)                    
             
        elif constraint == 'tpr_and_fpr':
            constraints_list = self.get_equal_tpr_and_fpr_constraints_dro(constraints_slack=constraints_slack)     
        
        else: 
            raise("constraint %s not supported for DRO." % (constraint))
        self.num_constraints = len(constraints_list)
        self.constraints = tf.convert_to_tensor(constraints_list)
        
        # Lagrangian loss to minimize
        lagrangian_loss = self.objective + tf.tensordot(
          tf.cast(self.lambda_variables, dtype=self.constraints.dtype.base_dtype),
          self.constraints, 1)

        optimizer_theta = tf.train.AdamOptimizer(learning_rate_theta)
        optimizer_lambda = tf.train.AdamOptimizer(learning_rate_lambda)
        optimizer_p_list = []
        for i in range(len(learning_rate_p_list)):
            #print('create optimizer p, ', i, learning_rate_p_list[i])
            optimizer_p_list.append(tf.train.AdamOptimizer(learning_rate_p_list[i]))
        self.train_op_theta = optimizer_theta.minimize(lagrangian_loss, var_list=self.theta_variables)
        self.train_op_lambda = optimizer_lambda.minimize(-lagrangian_loss, var_list=self.lambda_variables)
        self.train_op_p_list = []
        
       
        for i in range(self.num_groups):
           
            optimizer_p = optimizer_p_list[i]
            p_variable = self.p_variables_list[i]
            
            train_op_p = optimizer_p.minimize(-lagrangian_loss, var_list=p_variable)
            self.train_op_p_list.append(train_op_p)
       
        return self.train_op_theta, self.train_op_lambda, self.train_op_p_list

def training_generator(model,
                       train_df,
                       val_df,
                       test_df,
                       minibatch_size=None,
                       num_iterations_per_loop=1,
                       num_loops=1):
    tf.set_random_seed(31337)
    num_rows = train_df.shape[0]
    p_variables_list_all_loop = []
    
    if minibatch_size is None:
        print('minibatch is off')
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
        print('start loop ', n+1, 'in loops ', num_loops)
        loop_start_time = time.time()
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
            # Ascent step on p.
            for i in range(model.num_groups):
                session.run(
                      model.train_op_p_list[i],
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
        p_variables_list = session.run(model.p_variables_list)
        print('finish loop ', n+1, 'in loops ', num_loops)
        print('time for this loop ',time.time() - loop_start_time)

        yield (objective, constraints, train_predictions, lambda_variables, p_variables_list, val_predictions, test_predictions)
        

def training_helper(model,
                    train_df,
                    val_df,
                    test_df,
                    protected_columns, 
                    proxy_columns, 
                    label_column,
                    train_phats, val_phats, test_phats, 
                    constraint = 'tpr_and_fpr', 
                    minibatch_size = None,
                    num_iterations_per_loop=1,
                    num_loops=1, maximum_p_radius = 0.5, max_num_ptilde = 20, max_diff = [0.02, 0.02]):
    
    train_hinge_objective_vector = []
    train_01_objective_vector = [] # List of T scalar values representing the 01 objective at each iteration.
    #ranked by: train_01_objective_vector, s.t. train_01_proxy_Ghat_constraints_matrix
    #want to change to: train_01_objective_vector, s.t. train_01_robust_constraints_matrix
    train_01_true_G_constraints_matrix = [] # List of T vectors of size m, where each vector[i] is the zero-one constraint violation for group i.
    # Eventually we will just pick the last vector in this list, and take the max over m entries to get the max constraint violation.
    train_01_proxy_Ghat_constraints_matrix = []
    train_hinge_constraints_matrix = []  # Hinge loss constraint violations on the proxy groups.
    train_robust_constraints_matrix = []
 
    lambda_variables_matrix = []
    p_variables_list_matrix = []
    
    val_01_objective_vector = []
    val_01_true_G_constraints_matrix = []
    val_01_proxy_Ghat_constraints_matrix = []
    val_robust_constraints_matrix = []

    test_01_objective_vector = []
    test_01_true_G_constraints_matrix = []
    test_01_proxy_Ghat_constraints_matrix = []
    test_robust_constraints_matrix = []
   
    for objective, constraints, train_predictions, lambda_variables, p_variables_list, val_predictions, test_predictions in training_generator(
      model, train_df, val_df, test_df, minibatch_size, num_iterations_per_loop,
      num_loops):
        
        lambda_variables_matrix.append(lambda_variables)
        p_variables_list_matrix.append(p_variables_list)
        
        train_hinge_objective_vector.append(objective)
        train_hinge_constraints_matrix.append(constraints)
        train_df['predictions'] = train_predictions
        
        train_01_objective, train_01_true_G_constraints, train_01_proxy_Ghat_constraints = losses.get_error_rate_and_constraints(train_df, protected_columns, proxy_columns, label_column, max_diff = max_diff, constraint = constraint)
        
        train_01_objective_vector.append(train_01_objective)
        train_01_true_G_constraints_matrix.append(train_01_true_G_constraints)
        train_01_proxy_Ghat_constraints_matrix.append(train_01_proxy_Ghat_constraints)
       
        train_robust_constraints = get_robust_constraints(train_df, train_phats, proxy_columns,label_column, maximum_p_radius=maximum_p_radius, max_num_ptilde = max_num_ptilde, max_diff = max_diff, constraint = constraint)
        train_robust_constraints_matrix.append(train_robust_constraints)
        
        val_df['predictions'] = val_predictions
        val_01_objective, val_01_true_G_constraints, val_01_proxy_Ghat_constraints = losses.get_error_rate_and_constraints(val_df, protected_columns, proxy_columns, label_column)
        val_01_objective_vector.append(val_01_objective)
        val_01_true_G_constraints_matrix.append(val_01_true_G_constraints)
        val_01_proxy_Ghat_constraints_matrix.append(val_01_proxy_Ghat_constraints)
        
        val_robust_constraints = get_robust_constraints(val_df, val_phats, proxy_columns,label_column, maximum_p_radius=maximum_p_radius, max_num_ptilde = max_num_ptilde, max_diff = max_diff, constraint = constraint)
        val_robust_constraints_matrix.append(val_robust_constraints)
        
        test_df['predictions'] = test_predictions
        test_01_objective, test_01_true_G_constraints, test_01_proxy_Ghat_constraints = losses.get_error_rate_and_constraints(test_df, protected_columns, proxy_columns, label_column)
        test_01_objective_vector.append(test_01_objective)
        test_01_true_G_constraints_matrix.append(test_01_true_G_constraints)
        test_01_proxy_Ghat_constraints_matrix.append(test_01_proxy_Ghat_constraints)
        
        test_robust_constraints = get_robust_constraints(test_df, test_phats, proxy_columns,label_column, maximum_p_radius=maximum_p_radius, max_num_ptilde = max_num_ptilde, max_diff = max_diff, constraint = constraint)
        test_robust_constraints_matrix.append(test_robust_constraints)
       
    return {'train_hinge_objective_vector': train_hinge_objective_vector, 
            'train_hinge_constraints_matrix': train_hinge_constraints_matrix, 
            'train_01_objective_vector': train_01_objective_vector, 
            'train_01_true_G_constraints_matrix': train_01_true_G_constraints_matrix, 
            'train_01_proxy_Ghat_constraints_matrix': train_01_proxy_Ghat_constraints_matrix, 
            'train_robust_constraints_matrix': train_robust_constraints_matrix, 
            'lambda_variables_matrix': lambda_variables_matrix, 
            'p_variables_list_matrix': p_variables_list_matrix,
            'val_01_objective_vector': val_01_objective_vector, 
            'val_01_true_G_constraints_matrix': val_01_true_G_constraints_matrix, 
            'val_01_proxy_Ghat_constraints_matrix': val_01_proxy_Ghat_constraints_matrix,
            'val_robust_constraints_matrix': val_robust_constraints_matrix, 
            'test_01_objective_vector': test_01_objective_vector, 
            'test_01_true_G_constraints_matrix': test_01_true_G_constraints_matrix, 
            'test_01_proxy_Ghat_constraints_matrix': test_01_proxy_Ghat_constraints_matrix,
            'test_robust_constraints_matrix': test_robust_constraints_matrix}

    
def get_results_for_learning_rates(input_df, 
                                    feature_names, protected_columns, proxy_columns, label_column, 
                                    constraint = 'tpr_and_fpr', 
                                    learning_rates_theta = [0.001, 0.01, 0.1], #[0.001,0.01,0.1]
                                    learning_rates_lambda = [0.1, 0.5, 1, 2], # [0.5, 1, 2]
                                    learning_rate_p_lists = [[0.001, 0.001, 0.001],
                                                             [0.01, 0.01, 0.01],[0.1, 0.1, 0.1]], 
                                    num_runs=1,  #10, num of splits
                                    minibatch_size=None,  #1000
                                    num_iterations_per_loop=25,  #100
                                    num_loops=30, #30
                                    constraints_slack=1, maximum_p_radius = [1,1,1], max_num_ptilde = 20, max_diff = [0.02, 0.02]):  
    #generate learning rates for p
    #learning_rate_p_lists = list(itertools.product(*learning_rates_p_list))
    
    ts = time.time()
    # 10 runs with mean and stddev
    results_dicts_runs = []
    for i in range(num_runs):
        print('Split %d of %d' % (i+1, num_runs))
        t_split = time.time()
        train_df, val_df, test_df = data.train_val_test_split(input_df, 0.6, 0.2, seed=88+i)
        train_phats = data.compute_phats(train_df, proxy_columns)
        val_phats = data.compute_phats(val_df, proxy_columns)
        test_phats = data.compute_phats(test_df, proxy_columns)
        val_objectives = []
        val_constraints_matrix = []
        results_dicts = []
        learning_rates_iters_theta = []
        learning_rates_iters_lambda = []
        learning_rates_iters_p_list = []
        for learning_rate_p_list in learning_rate_p_lists:
            for learning_rate_theta in learning_rates_theta:
                for learning_rate_lambda in learning_rates_lambda:
                    t_start_iter = time.time() - ts
                    print("time since start:", t_start_iter)
                    print("begin optimizing learning rate p list:", learning_rate_p_list)
                    print("begin optimizing learning rate theta: %.3f learning rate lambda: %.3f" % (learning_rate_theta, learning_rate_lambda))
                   
                
                    model = DRO_Model(feature_names, proxy_columns, label_column, train_phats, maximum_lambda_radius=1, maximum_p_radius=maximum_p_radius)
                  
                    model.build_train_ops(constraint=constraint, learning_rate_theta=learning_rate_theta, learning_rate_lambda=learning_rate_lambda,learning_rate_p_list = learning_rate_p_list, constraints_slack=constraints_slack)
                    # training_helper returns the list of errors and violations over each epoch. 
                    results_dict = training_helper(
                          model,
                          train_df,
                          val_df,
                          test_df,
                          protected_columns, 
                          proxy_columns, 
                          label_column,
                          train_phats, val_phats, test_phats, 
                          constraint = constraint,
                          minibatch_size=minibatch_size,
                          num_iterations_per_loop=num_iterations_per_loop,
                          num_loops=num_loops, maximum_p_radius=maximum_p_radius, max_num_ptilde=max_num_ptilde, max_diff = max_diff)

                    #find index for the best train iteration for this pair of hyper parameters
                    best_index_iters = utils.find_best_candidate_index(np.array(results_dict['train_01_objective_vector']),np.array(results_dict['train_robust_constraints_matrix']))
                    val_objectives.append(results_dict['val_01_objective_vector'][best_index_iters])
                    val_constraints_matrix.append(results_dict['val_robust_constraints_matrix'][best_index_iters])
                    results_dict_best_idx = utils.add_results_dict_best_idx_robust(results_dict, best_index_iters)
                    results_dicts.append(results_dict_best_idx)
                    learning_rates_iters_theta.append(learning_rate_theta)
                    learning_rates_iters_lambda.append(learning_rate_lambda)
                    learning_rates_iters_p_list.append(learning_rate_p_list)
                    print("Finished learning rate p list", learning_rate_p_list)
                    print("Finished optimizing learning rate theta: %.3f learning rate lambda: %.3f " % (learning_rate_theta, learning_rate_lambda))
                    print("Time that this run took:", time.time() - t_start_iter - ts)
        #find the index of the best pair of hyper parameters        
        best_index = utils.find_best_candidate_index(np.array(val_objectives),np.array(val_constraints_matrix))
        best_results_dict = results_dicts[best_index]
        best_learning_rate_theta = learning_rates_iters_theta[best_index]
        best_learning_rate_lambda = learning_rates_iters_lambda[best_index]
        best_learning_rate_p_list = learning_rates_iters_p_list[best_index]
        print('best_learning_rate_theta,', best_learning_rate_theta)
        print('best_learning_rate_lambda', best_learning_rate_lambda)
        print('best_learning_rate_p_list', best_learning_rate_p_list)
        results_dicts_runs.append(best_results_dict)
        print("time it took for this split", time.time() - t_split)
        
    final_average_results_dict = utils.average_results_dict_fn(results_dicts_runs)
    
    return final_average_results_dict

def get_robust_constraints(df, phats, proxy_columns, label_column, max_diff=[0.05, 0.05], max_num_ptilde = 20, maximum_p_radius=[1,1,1], constraint = 'tpr_and_fpr'):
    """Computes the robust fairness violations.
    
    Args:
      df: dataframe containing 'predictions' column and LABEL_COLUMN, PROXY_COLUMNS.
        predictions column is not required to be thresholded.
    
    """
    if constraint == 'tpr':
        tpr_overall = losses.tpr(df, label_column)
        print('tpr_overall', tpr_overall)
        robust_constraints = []
        for i in range(len(proxy_columns)):
            robust_constraint = -5
            for j in range(max_num_ptilde):
                ptilde = utils.generate_rand_vec_l1_ball(phats[i], maximum_p_radius[i])
                labels = np.array(df[label_column] > 0.5)

                tp = np.array((df['predictions'] >= 0.0) & (df[label_column] > 0.5))
                weighted_labels = np.multiply(ptilde, labels)
                weighted_tp = np.multiply(ptilde, tp)
                weighted_tpr = float(sum(weighted_tp)/sum(weighted_labels))
                new_robust_constraint = tpr_overall - weighted_tpr - max_diff
                if new_robust_constraint > robust_constraint:
                    robust_constraint = new_robust_constraint        
            robust_constraints.append(robust_constraint)
    
    elif constraint == 'tpr_and_fpr':
        tpr_overall = losses.tpr(df, label_column)
        print('tpr_overall', tpr_overall)
        fpr_overall = losses.fpr(df, label_column)
        print('fpr_overall', fpr_overall)
        robust_constraints_tpr = []
        robust_constraints_fpr = []
        for i in range(len(proxy_columns)):
            robust_constraint_tpr = -5
            robust_constraint_fpr = -5
            for j in range(max_num_ptilde):
                ptilde = utils.generate_rand_vec_l1_ball(phats[i], maximum_p_radius[i])
                labels = np.array(df[label_column] > 0.5)

                tp = np.array((df['predictions'] >= 0.0) & (df[label_column] > 0.5))
                weighted_labels = np.multiply(ptilde, labels)
                weighted_tp = np.multiply(ptilde, tp)
                weighted_tpr = float(sum(weighted_tp)/sum(weighted_labels))
                new_robust_constraint_tpr = tpr_overall - weighted_tpr - max_diff[0]
                if new_robust_constraint_tpr > robust_constraint_tpr:
                    robust_constraint_tpr = new_robust_constraint_tpr 
           
                fp = np.array((df['predictions'] >= 0.0) & (df[label_column] <= 0.5))
                weighted_flipped_labels = np.multiply(ptilde, np.array(df[label_column] <= 0.5))
                weighted_fp = np.multiply(ptilde, fp)
                weighted_fpr = float(sum(weighted_fp)/sum(weighted_flipped_labels))
                new_robust_constraint_fpr = weighted_fpr - fpr_overall - max_diff[1]
                if new_robust_constraint_fpr > robust_constraint_fpr:
                    robust_constraint_fpr = new_robust_constraint_fpr 
                
            robust_constraints_tpr.append(robust_constraint_tpr)
            robust_constraints_fpr.append(robust_constraint_fpr)
        robust_constraints = robust_constraints_tpr + robust_constraints_fpr
      
    return robust_constraints
        
# Expects averaged results_dict with means and standard deviations.
def plot_optimization_avg(results_dict, protected_columns, proxy_columns):
    fig, axs = plt.subplots(5, figsize=(5,25))
    num_iters = len(results_dict['train_hinge_objective_vector'][0])
    iters = np.arange(num_iters)
    axs[0].errorbar(iters, results_dict['train_hinge_objective_vector'][0], yerr=results_dict['train_hinge_objective_vector'][1])
    axs[0].set_title('train_hinge_objective_vector')
    for i in range(len(protected_columns)):
        axs[1].errorbar(iters, results_dict['train_hinge_constraints_matrix'][0].T[i], results_dict['train_hinge_constraints_matrix'][1].T[i], label=protected_columns[i])
        axs[2].errorbar(iters, results_dict['train_01_proxy_Ghat_constraints_matrix'][0].T[i], results_dict['train_01_proxy_Ghat_constraints_matrix'][1].T[i], label=proxy_columns[i])
        axs[3].errorbar(iters, results_dict['train_01_true_G_constraints_matrix'][0].T[i], results_dict['train_01_true_G_constraints_matrix'][1].T[i], label=protected_columns[i])
        axs[4].errorbar(iters, results_dict['train_robust_constraints_matrix'][0].T[i], results_dict['train_robust_constraints_matrix'][1].T[i], label=protected_columns[i])
    axs[1].set_title('train_hinge_constraints_matrix')
    axs[1].legend()
    axs[2].set_title('train_01_proxy_Ghat_constraints_matrix')
    axs[2].legend()
    axs[3].set_title('train_01_true_G_constraints_matrix')
    axs[3].legend()
    axs[4].set_title('train_robust_constraints_matrix')
    axs[4].legend()
    plt.show()
    
# Expects results dicts without averaging.
def plot_optimization_dro(results_dict):
    fig, axs = plt.subplots(5, figsize=(5,25))
    axs[0].plot(results_dict['train_hinge_objective_vector'])
    axs[0].set_title('train_hinge_objective_vector')
    axs[1].plot(results_dict['train_hinge_constraints_matrix'])
    axs[1].set_title('train_hinge_constraints_matrix')
    axs[2].plot(results_dict['train_01_proxy_Ghat_constraints_matrix'])
    axs[2].set_title('train_01_proxy_Ghat_constraints_matrix')
    axs[3].plot(results_dict['train_01_true_G_constraints_matrix'])
    axs[3].set_title('train_01_true_G_constraints_matrix')
    axs[4].plot(results_dict['train_robust_constraints_matrix'])
    axs[4].set_title('train_robust_constraints_matrix')
    plt.show()

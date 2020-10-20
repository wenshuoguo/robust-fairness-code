""" Functions for training using the soft assignments approach."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.optimize import linprog
import tensorflow as tf
import time

import data
import losses
import optimization
import model
import utils


class SoftweightsHeuristicModel(model.Model):
    """Linear model for performing constrained optimization with soft assignments practical algorithm.
        
    Args:
      b: numpy array of floats of length (num_groups^2 + 4*num_groups). 
        First m^2 entries are prior values for P(G = j | \hat{G} = k). Last 4*num_groups entries are 1s.
      feature_names: list of strings, a list of names of all feature columns.
      protected_columns: list of strings, a list of the names of all protected group columns 
        (column should contain values of 0 or 1 representing group membership).
      label_column: string, name of label column. Column should contain values of 0 or 1.
      maximum_lambda_radius: float, an optional upper bound to impose on the
        sum of the lambdas.
    
    Raises:
      ValueError: if "maximum_lambda_radius" is nonpositive.  
    """   
    def __init__(self, b, true_group_marginals, feature_names, protected_columns, label_column, maximum_lambda_radius=None):
        super().__init__(feature_names, protected_columns, label_column, maximum_lambda_radius=maximum_lambda_radius)
        self.W_num_rows = 4*self.num_groups
        self.W_num_cols = self.num_groups
        self.W_flattened_size = self.W_num_rows*self.W_num_cols
        self.b_groups = tf.convert_to_tensor(b, dtype=tf.float32)
        self.b_groups = tf.reshape(self.b_groups, [-1,1])
        self.b_simplex = tf.ones((self.W_num_rows,1))
        self.true_group_marginals = tf.convert_to_tensor(true_group_marginals, dtype=tf.float32)
        
    
    # Builds r tensor, convexified for the particular l_1 = indicator(prediction*label <= 0).
    def build_r_tensor_err(self):
        r_list = []
        for protected_placeholder in self.protected_placeholders:
            # r[4k-3]: group = k, prediction = 0, label = 0
            r_list.append(0) 
            # r[4k-2]: group = k, prediction = 0, label = 1
            r_list.append(tf.losses.hinge_loss(self.labels_placeholder, self.predictions_tensor, weights=tf.multiply(protected_placeholder,self.labels_placeholder), reduction=tf.compat.v1.losses.Reduction.SUM))
            # r[4k-1]: group = k, prediction = 1, label = 0
            r_list.append(tf.losses.hinge_loss(self.labels_placeholder, self.predictions_tensor, weights=tf.multiply(protected_placeholder,utils.flip_binary_tensor(self.labels_placeholder)), reduction=tf.compat.v1.losses.Reduction.SUM))
            # r[4k]: group = k, prediction = 1, label = 1
            r_list.append(0) 
        assert(len(r_list) == (self.W_num_rows))
        self.r_tensor = tf.convert_to_tensor(r_list)
        
        
    def build_r_tensor_tpr(self):
        # Build r_1 (numerator r)
        r_list_1 = []
        for protected_placeholder in self.protected_placeholders:
            # r[4k-3]: group = k, prediction = 0, label = 0
            r_list_1.append(0) 
            # r[4k-2]: group = k, prediction = 0, label = 1
            r_list_1.append(0) 
            # r[4k-1]: group = k, prediction = 1, label = 0
            r_list_1.append(0) 
            # r[4k]: group = k, prediction = 1, label = 1
            r_list_1.append(losses.concave_hinge_loss( 
                utils.flip_binary_tensor(self.labels_placeholder), 
                self.predictions_tensor, 
                weights=tf.multiply(protected_placeholder,self.labels_placeholder), 
                reduction=tf.compat.v1.losses.Reduction.SUM)) 
        assert(len(r_list_1) == (self.W_num_rows))
        self.r_tensor_1_tpr = tf.convert_to_tensor(r_list_1)

        # Build r_2 (denominator r)
        r_list_2 = []
        for protected_placeholder in self.protected_placeholders:
            # r[4k-3]: group = k, prediction = 0, label = 0
            r_list_2.append(0) 
            # r[4k-2]: group = k, prediction = 0, label = 1
            r_list_2.append(losses.ramp_loss(
                self.labels_placeholder, 
                self.predictions_tensor, 
                weights=tf.multiply(protected_placeholder,self.labels_placeholder), 
                reduction=tf.compat.v1.losses.Reduction.SUM)) 
            # r[4k-1]: group = k, prediction = 1, label = 0
            r_list_2.append(0) 
            # r[4k]: group = k, prediction = 1, label = 1
            r_list_2.append(losses.ramp_loss(
                utils.flip_binary_tensor(self.labels_placeholder), 
                self.predictions_tensor, 
                weights=tf.multiply(protected_placeholder,self.labels_placeholder), 
                reduction=tf.compat.v1.losses.Reduction.SUM)) 
        assert(len(r_list_2) == (self.W_num_rows))
        self.r_tensor_2_tpr = tf.convert_to_tensor(r_list_2)


    def build_r_tensor_fpr(self):
        # Build r_1 (numerator r)
        r_list_1 = []
        for protected_placeholder in self.protected_placeholders:
            # r[4k-3]: group = k, prediction = 0, label = 0
            r_list_1.append(0) 
            # r[4k-2]: group = k, prediction = 0, label = 1
            r_list_1.append(0)
            # r[4k-1]: group = k, prediction = 1, label = 0
            r_list_1.append(tf.losses.hinge_loss(
                self.labels_placeholder, 
                self.predictions_tensor, 
                weights=tf.multiply(protected_placeholder,utils.flip_binary_tensor(self.labels_placeholder)), 
                reduction=tf.compat.v1.losses.Reduction.SUM))
            # r[4k]: group = k, prediction = 1, label = 1
            r_list_1.append(0) 
        assert(len(r_list_1) == (self.W_num_rows))
        self.r_tensor_1_fpr = tf.convert_to_tensor(r_list_1)

        # Build r_2 (denominator r)
        r_list_2 = []
        for protected_placeholder in self.protected_placeholders:
            # r[4k-3]: group = k, prediction = 0, label = 0
            r_list_2.append(losses.ramp_loss(
                utils.flip_binary_tensor(self.labels_placeholder), 
                self.predictions_tensor, 
                weights=tf.multiply(protected_placeholder,utils.flip_binary_tensor(self.labels_placeholder)), 
                reduction=tf.compat.v1.losses.Reduction.SUM)) 
            # r[4k-2]: group = k, prediction = 0, label = 1
            r_list_2.append(0)
            # r[4k-1]: group = k, prediction = 1, label = 0
            r_list_2.append(losses.ramp_loss(
                self.labels_placeholder, 
                self.predictions_tensor, 
                weights=tf.multiply(protected_placeholder,utils.flip_binary_tensor(self.labels_placeholder)), 
                reduction=tf.compat.v1.losses.Reduction.SUM)) 
            # r[4k]: group = k, prediction = 1, label = 1
            r_list_2.append(0) 
        assert(len(r_list_2) == (self.W_num_rows))
        self.r_tensor_2_fpr = tf.convert_to_tensor(r_list_2)
    
    
    def build_r_tensor(self, constraint='tpr'):
        if constraint == 'err':
            self.build_r_tensor_err()
        elif constraint == 'tpr':
            self.build_r_tensor_tpr()
        elif constraint == 'tpr_and_fpr':
            self.build_r_tensor_tpr()
            self.build_r_tensor_fpr()
        else:
            raise('constraint not recognized.')
    
    
    def build_v_tensors(self):
        # Update v according to the batch.
        # First construct v as a dense vector in R^4m.
        v_list = []
        # Compute thresholded predictions.
        thresholded_predictions = tf.dtypes.cast(tf.math.greater(self.predictions_tensor, 0.0), tf.float32)
        for protected_placeholder in self.protected_placeholders:
            group_size = tf.reduce_sum(protected_placeholder)
            # v[4k-3]: group = k, prediction = 0, label = 0
            v_list.append(tf.math.divide_no_nan(tf.reduce_sum(tf.multiply(protected_placeholder, tf.multiply(utils.flip_binary_tensor(thresholded_predictions), utils.flip_binary_tensor(self.labels_placeholder)))), 
                                         group_size))
            # v[4k-2]: group = k, prediction = 0, label = 1
            v_list.append(tf.math.divide_no_nan(tf.reduce_sum(tf.multiply(protected_placeholder, tf.multiply(utils.flip_binary_tensor(thresholded_predictions), self.labels_placeholder))), 
                                         group_size))
            # v[4k-1]: group = k, prediction = 1, label = 0
            v_list.append(tf.math.divide_no_nan(tf.reduce_sum(tf.multiply(protected_placeholder, tf.multiply(thresholded_predictions, utils.flip_binary_tensor(self.labels_placeholder)))), 
                                         group_size))
            # v[4k]: group = k, prediction = 1, label = 1
            v_list.append(tf.math.divide_no_nan(tf.reduce_sum(tf.multiply(protected_placeholder, tf.multiply(thresholded_predictions, self.labels_placeholder))), 
                                         group_size))
        full_v_tensor = tf.convert_to_tensor(v_list)
        self.full_v_tensor = full_v_tensor
        # Mask v_list to create m different vectors in R^4m.
        self.v_tensors = []
        for j in range(self.num_groups):
            start = j*4
            mask_tensor = utils.binary_mask_tensor(self.W_num_rows, start, 4)
            new_v_tensor = tf.multiply(mask_tensor, full_v_tensor)
            self.v_tensors.append(new_v_tensor)
    
    
    def get_flattened_v(self, v, j):
        """Creates a vector in R^{4m^2} such that when multiplied by flattened W, the result is v^TWe_j.
        
        Args:
          v: tensor in R^4m.
          j: int, zero indexed column number of W.
        """ 
        front_padding = self.W_num_rows * j
        back_padding = self.W_flattened_size - (front_padding + self.W_num_rows)
        # Set indices equivalent to jth column of W to v.
        paddings = tf.constant([[front_padding, back_padding]])
        flattened_v = tf.pad(v, paddings, "CONSTANT")
        return flattened_v
    
    
    def get_ones_for_row(self, i):
        """Creates a vector in R^{4m^2} such that when multiplied by flattened W, 
        the result is e_i^TW1 (multiplying the ith row of W by a vector of 1s in R^m).
        
        Args:
          i: int, zero indexed row number of W.
        """
        flattened_ones = np.zeros((self.W_flattened_size,))
        for j in range(self.num_groups):
            flattened_ones[j*self.W_num_rows + i] = 1
        return tf.convert_to_tensor(flattened_ones, dtype=tf.float32)
    

    def build_A_groups(self):
        # Create v tensors for projection.
        self.build_v_tensors() 
        A_groups = []
        # Add constraints related to v
        for j in range(self.num_groups):
            for k in range(self.num_groups):
                # By convention, the inner loop k represents the proxy group.
                A_groups.append(self.get_flattened_v(self.v_tensors[k], j))
        self.A_groups = tf.convert_to_tensor(A_groups)
    
    
    def build_A_simplex(self):
        # Add constraints related to simplex constraint.
        A_simplex = []
        for i in range(self.W_num_rows):
            A_simplex.append(self.get_ones_for_row(i))
        self.A_simplex = tf.convert_to_tensor(A_simplex)
        
        
    def project_W_linear_equality(self, W_input, A, b):
        """Projects W onto linear equality constraints AW_flattened = b."""
        # Flatten W into a column vector such that the first 4m elements of the flattened vector correspond with the first column of W.
        W_flattened = tf.reshape(tf.transpose(W_input), [self.W_flattened_size,1])
        # Project W_flattened onto linear constraints using A. 
        AT = tf.transpose(A)
        AATinv = tf.linalg.inv(tf.linalg.matmul(A, AT))
        AW_minus_b = tf.matmul(A, W_flattened) - b
        W_flattened_projected = W_flattened - tf.linalg.matmul(AT, tf.linalg.matmul(AATinv, AW_minus_b))
        # Unflatten back into matrix.
        W_projected = tf.transpose(tf.reshape(W_flattened_projected, [self.W_num_cols, self.W_num_rows]))
        return W_projected
    
    
    def project_W_groups(self, W_input):
        """Projects W onto linear equality constraints corresponding to v_k^TWe_j = b_groups_jk."""
        return self.project_W_linear_equality(W_input, self.A_groups, self.b_groups)
        
        
    def project_W_simplex(self, W_input):
        """Projects W onto linear equality constraints corresponding to W1=1."""
        return self.project_W_linear_equality(W_input, self.A_simplex, self.b_simplex)
    
    
    def project_W(self, W_input):
        """Projects W onto both linear equality constraints and W >= 0."""
        return optimization.project_by_dykstra(W_input, self.project_W_groups, self.project_W_simplex, num_iterations=self.num_projection_iters)
    
    
    def get_equal_accuracy_constraints(self, constraints_slack=1.0):
        constraints_list = []
        average_concave_hinge = losses.concave_hinge_loss(self.labels_placeholder, self.predictions_tensor)
        for j in range(self.num_groups):
            # Compute r^T W e_j
            W_j = tf.gather(self.W_variable, j, axis=1)
            Wterm = tf.tensordot(self.r_tensor,W_j, 1)
            # divide by batch size * P(G = j) (true G here)
            Wterm = tf.math.divide(Wterm, self.true_group_marginals[j] * tf.cast(tf.size(self.labels_placeholder), tf.float32))
            constraints_list.append(Wterm - average_concave_hinge - (constraints_slack * tf.ones_like(average_concave_hinge)))
        return constraints_list
    
    
    def get_equal_tpr_constraints(self, constraints_slack=1.0):
        constraints_list = []
        average_tpr = tf.losses.hinge_loss(utils.flip_binary_tensor(self.labels_placeholder), self.predictions_tensor, weights=self.labels_placeholder, reduction=tf.compat.v1.losses.Reduction.MEAN)
        for j in range(self.num_groups):
            # Compute r^T W e_j
            W_j = tf.gather(self.W_variable, j, axis=1)
            Wterm_numerator = tf.tensordot(self.r_tensor_1_tpr, W_j, 1) 
            Wterm_denominator = tf.tensordot(self.r_tensor_2_tpr, W_j, 1) + tf.ones_like(Wterm_numerator) # Include + 1 here to prevent denominator from being 0.
            # divide
            Wterm = tf.math.divide(Wterm_numerator, Wterm_denominator)
            constraints_list.append(average_tpr - Wterm - (constraints_slack * tf.ones_like(average_tpr)))
        return constraints_list


    def get_equal_fpr_constraints(self, constraints_slack=1.0):
        constraints_list = []
        average_fpr = losses.concave_hinge_loss(self.labels_placeholder, self.predictions_tensor, weights=utils.flip_binary_tensor(self.labels_placeholder), reduction=tf.compat.v1.losses.Reduction.MEAN)
        for j in range(self.num_groups):
            # Compute r^T W e_j
            W_j = tf.gather(self.W_variable, j, axis=1)
            Wterm_numerator = tf.tensordot(self.r_tensor_1_fpr, W_j, 1) 
            Wterm_denominator = tf.tensordot(self.r_tensor_2_fpr, W_j, 1) + tf.ones_like(Wterm_numerator) # Include + 1 here to prevent denominator from being 0.
            # divide
            Wterm = tf.math.divide(Wterm_numerator, Wterm_denominator)
            constraints_list.append(Wterm - average_fpr - (constraints_slack * tf.ones_like(average_fpr)))
        return constraints_list


    def get_equal_tpr_and_fpr_constraints(self, constraints_slack=1.0):
        equal_tpr_constraints = self.get_equal_tpr_constraints(constraints_slack)
        equal_fpr_constraints = self.get_equal_fpr_constraints(constraints_slack)
        constraints_list = equal_tpr_constraints + equal_fpr_constraints
        return constraints_list
    
    
    def build_train_ops(self, constraint='tpr', learning_rate_theta=0.01, learning_rate_lambda=0.01, 
                        learning_rate_W=0.01, constraints_slack=1.0, num_projection_iters=20):
        """Builds operators that take gradient steps during training.
        
        Args: 
          learning_rate_theta: float, learning rate for theta parameter on descent step.
          learning_rate_lambda: float, learning rate for lambda parameter on ascent step.
          constraints_slack: float, amount of slack for constraints. New constraint will be
              original_constraint - constraints_slack
        
        """
        # Hinge loss objective.
        self.objective = tf.losses.hinge_loss(self.labels_placeholder, self.predictions_tensor)
       
        # Create A matrix for projection.
        self.build_A_groups()
        self.build_A_simplex()
        
        # Create W variable.
        self.num_projection_iters=num_projection_iters
        initial_W = np.zeros((self.W_num_rows, self.W_num_cols), dtype=np.float32)
        self.W_variable = tf.compat.v2.Variable(
          initial_W,
          trainable=True,
          name="W",
          dtype=tf.float32,
          constraint=self.project_W
        )
        
        # Build constraints list for hinge loss equal accuracy constraint.
        self.build_r_tensor(constraint=constraint)
        constraints_list = []
        if constraint == 'err':
            constraints_list = self.get_equal_accuracy_constraints(constraints_slack=constraints_slack)
        elif constraint == 'tpr':
            constraints_list = self.get_equal_tpr_constraints(constraints_slack=constraints_slack)
        elif constraint == 'tpr_and_fpr':
            constraints_list = self.get_equal_tpr_and_fpr_constraints(constraints_slack=constraints_slack)
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
        optimizer_W = tf.train.AdamOptimizer(learning_rate_W)

        self.train_op_theta = optimizer_theta.minimize(lagrangian_loss, var_list=self.theta_variables)
        self.train_op_lambda = optimizer_lambda.minimize(-lagrangian_loss, var_list=self.lambda_variables)
        self.train_op_W = optimizer_W.minimize(-lagrangian_loss, var_list=self.W_variable)
        return self.train_op_theta, self.train_op_lambda, self.train_op_W


def training_generator(sw_model,
                       train_df,
                       val_df,
                       test_df,
                       minibatch_size=None,
                       num_iterations_per_loop=1,
                       num_loops=1,
                       num_iterations_W=1):
    tf.set_random_seed(31337)
    num_rows = train_df.shape[0]
    if minibatch_size is None:
        minibatch_size = num_rows
    else:
        minibatch_size = min(minibatch_size, num_rows)
    permutation = list(range(train_df.shape[0]))
    random.seed(88)
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
            # Ascent step on W (with projection included).
            for _ in range(num_iterations_W):
                session.run(
                      sw_model.train_op_W,
                      feed_dict=sw_model.feed_dict_helper(minibatch_df))
            # Descent step on theta.
            session.run(
                  sw_model.train_op_theta,
                  feed_dict=sw_model.feed_dict_helper(minibatch_df))
            # Ascent step on lambda (with projection included).
            session.run(
                  sw_model.train_op_lambda,
                  feed_dict=sw_model.feed_dict_helper(minibatch_df))

        objective = session.run(sw_model.objective, sw_model.feed_dict_helper(train_df))
        constraints = session.run(sw_model.constraints, sw_model.feed_dict_helper(train_df))
        train_predictions = session.run(
            sw_model.predictions_tensor,
            feed_dict=sw_model.feed_dict_helper(train_df))
        val_predictions = session.run(
            sw_model.predictions_tensor,
            feed_dict=sw_model.feed_dict_helper(val_df))
        test_predictions = session.run(
            sw_model.predictions_tensor,
            feed_dict=sw_model.feed_dict_helper(test_df))
        lambda_variables = session.run(sw_model.lambda_variables)
        W_variable = session.run(sw_model.W_variable)

        yield (objective, constraints, train_predictions, lambda_variables, W_variable, val_predictions, test_predictions)


def get_r_from_data_tpr(df, proxy_columns, label_column):
    # r for numerator
    r_list_1 = []
    label_marginal = np.mean(df[label_column])
    for proxy_column in proxy_columns:
        # r[4k-3]: group = k, prediction = 0, label = 0
        r_list_1.append(0) 
        # r[4k-2]: group = k, prediction = 0, label = 1
        r_list_1.append(0) 
        # r[4k-1]: group = k, prediction = 1, label = 0
        r_list_1.append(0) 
        # r[4k]: group = k, prediction = 1, label = 1
        thresholded_predictions = (df['predictions'] > 0).astype(np.float32)
        indicators = np.multiply(thresholded_predictions, np.multiply(df[proxy_column], df[label_column]))
        r_list_1.append(np.sum(indicators)) 
    r_list_1_array = np.array(r_list_1)

    # r for denominator
    r_list_2 = []
    label_marginal = np.mean(df[label_column])
    for proxy_column in proxy_columns:
        # r[4k-3]: group = k, prediction = 0, label = 0
        r_list_2.append(0) 
        # r[4k-2]: group = k, prediction = 0, label = 1
        thresholded_predictions = (df['predictions'] <= 0).astype(np.float32)
        indicators = np.multiply(thresholded_predictions, np.multiply(df[proxy_column], df[label_column]))
        r_list_2.append(np.sum(indicators)) 
        # r[4k-1]: group = k, prediction = 1, label = 0
        r_list_2.append(0) 
        # r[4k]: group = k, prediction = 1, label = 1
        thresholded_predictions = (df['predictions'] > 0).astype(np.float32)
        indicators = np.multiply(thresholded_predictions, np.multiply(df[proxy_column], df[label_column]))
        r_list_2.append(np.sum(indicators)) 
    r_list_2_array = np.array(r_list_2)
    return r_list_1_array, r_list_2_array


def get_r_from_data_fpr(df, proxy_columns, label_column):
    # r for numerator
    r_list_1 = []
    label_marginal = np.mean(df[label_column])
    labels_array=np.array(df[label_column])
    for proxy_column in proxy_columns:
        # r[4k-3]: group = k, prediction = 0, label = 0
        r_list_1.append(0) 
        # r[4k-2]: group = k, prediction = 0, label = 1
        r_list_1.append(0) 
        # r[4k-1]: group = k, prediction = 1, label = 0
        thresholded_predictions = (df['predictions'] > 0).astype(np.float32)
        indicators = np.multiply(thresholded_predictions, np.multiply(df[proxy_column], utils.flip_binary_array(labels_array)))
        r_list_1.append(np.sum(indicators)) 
        # r[4k]: group = k, prediction = 1, label = 1
        r_list_1.append(0)  
    r_list_1_array = np.array(r_list_1)

    # r for denominator
    r_list_2 = []
    for proxy_column in proxy_columns:
        # r[4k-3]: group = k, prediction = 0, label = 0
        thresholded_predictions = (df['predictions'] <= 0).astype(np.float32)
        indicators = np.multiply(thresholded_predictions, np.multiply(df[proxy_column], utils.flip_binary_array(labels_array)))
        r_list_2.append(np.sum(indicators)) 
        # r[4k-2]: group = k, prediction = 0, label = 1
        r_list_2.append(0) 
        # r[4k-1]: group = k, prediction = 1, label = 0
        thresholded_predictions = (df['predictions'] > 0).astype(np.float32)
        indicators = np.multiply(thresholded_predictions, np.multiply(df[proxy_column], utils.flip_binary_array(labels_array)))
        r_list_2.append(np.sum(indicators)) 
        # r[4k]: group = k, prediction = 1, label = 1
        r_list_2.append(0) 
    r_list_2_array = np.array(r_list_2)
    return r_list_1_array, r_list_2_array


def get_r_from_data_err(df, proxy_columns, label_column):
    r_list = []
    labels_array=np.array(df[label_column])
    protected_array = np.array(df[proxy_column])
    thresholded_predictions = np.array((df['predictions'] > 0).astype(np.float32))
    for proxy_column in proxy_columns:
        # r[4k-3]: group = k, prediction = 0, label = 0
        r_list.append(0) 
        # r[4k-2]: group = k, prediction = 0, label = 1
        indicators = np.multiply(utils.flip_binary_array(thresholded_predictions), np.multiply(protected_array, labels_array))
        r_list.append(np.sum(indicators)) 
        # r[4k-1]: group = k, prediction = 1, label = 0
        indicators = np.multiply(thresholded_predictions, np.multiply(protected_array, utils.flip_binary_array(labels_array)))
        r_list.append(np.sum(indicators)) 
        # r[4k]: group = k, prediction = 1, label = 1
        r_list.append(0) 
    return np.array(r_list)


def get_v_arrays_from_data(df, proxy_columns, label_column):
    # First construct v as a dense vector in R^4m.
    v_list = []
    # Compute thresholded predictions.
    thresholded_predictions = np.array((df['predictions'] > 0).astype(np.float32))
    labels_array = np.array(df[label_column])
    for proxy_column in proxy_columns:
        group_size = df[proxy_column].sum()
        # v[4k-3]: group = k, prediction = 0, label = 0
        numerator = np.sum(np.multiply(df[proxy_column], np.multiply(utils.flip_binary_array(thresholded_predictions), utils.flip_binary_array(labels_array))))
        v_list.append(np.nan_to_num(numerator/group_size))
        # v[4k-2]: group = k, prediction = 0, label = 1
        numerator = np.sum(np.multiply(df[proxy_column], np.multiply(utils.flip_binary_array(thresholded_predictions), labels_array)))
        v_list.append(np.nan_to_num(numerator/group_size))
        # v[4k-1]: group = k, prediction = 1, label = 0
        numerator = np.sum(np.multiply(df[proxy_column], np.multiply(thresholded_predictions, utils.flip_binary_array(labels_array))))
        v_list.append(np.nan_to_num(numerator/group_size))
        # v[4k]: group = k, prediction = 1, label = 1
        numerator = np.sum(np.multiply(df[proxy_column], np.multiply(thresholded_predictions, labels_array)))
        v_list.append(np.nan_to_num(numerator/group_size))
    full_v_array = np.array(v_list)
    # Mask v_list to create m different vectors in R^4m.
    num_groups = len(proxy_columns)
    W_num_rows = 4*num_groups
    # W_num_cols = num_groups
    v_arrays = []
    for j in range(num_groups):
        start = j*4
        mask_array = utils.binary_mask_array(W_num_rows, start, 4)
        new_v_array = np.multiply(mask_array, full_v_array)
        v_arrays.append(new_v_array)
    return v_arrays
    
    
def get_flattened_v_array(v, j, num_groups):
    """Creates a vector in R^{4m^2} such that when multiplied by flattened W, the result is v^TWe_j.
    
    Args:
      v: tensor in R^4m.
      j: int, zero indexed column number of W.
    """ 
    W_num_rows = 4*num_groups
    W_num_cols = num_groups
    W_flattened_size = W_num_rows * W_num_cols
    front_padding = W_num_rows * j
    back_padding = W_flattened_size - (front_padding + W_num_rows)
    # Set indices equivalent to jth column of W to v.
    paddings = (front_padding, back_padding)
    flattened_v = np.pad(v, paddings, "constant")
    return flattened_v
    

def get_ones_for_row_array(i, num_groups):
    """Creates a vector in R^{4m^2} such that when multiplied by flattened W, 
    the result is e_i^TW1 (multiplying the ith row of W by a vector of 1s in R^m).
    
    Args:
      i: int, zero indexed row number of W.
    """
    W_num_rows = 4*num_groups
    W_num_cols = num_groups
    W_flattened_size = W_num_rows * W_num_cols
    flattened_ones = np.zeros((W_flattened_size,))
    for j in range(num_groups):
        flattened_ones[j*W_num_rows + i] = 1
    return flattened_ones


def build_A_eq_array(df, proxy_columns, label_column, v_arrays):
    # Create v tensors for projection.
    num_groups = len(proxy_columns)
    A_eq = []
    # Add constraints related to v.
    for j in range(num_groups):
        for k in range(num_groups):
            # By convention, the inner loop k represents the proxy group.
            A_eq.append(get_flattened_v_array(v_arrays[k], j, num_groups))
    # Add simplex constraints.
    W_num_rows = 4*num_groups
    for i in range(W_num_rows):
        A_eq.append(get_ones_for_row_array(i, num_groups))
    A_eq = np.array(A_eq)
    return A_eq


def get_optimized_robust_constraints(df, proxy_columns, protected_columns, label_column, true_group_marginals, max_diff=0.05):
    """Computes robust constraints by explicitly maximizing over W."""
    raise("Doesn't currently work. Not updated.")
    r_array = get_r_from_data_tpr(df, proxy_columns, label_column)
    tpr_overall = losses.tpr(df, label_column)
    num_groups = len(proxy_columns)
    
    robust_constraints = []

    v_arrays = get_v_arrays_from_data(df, proxy_columns, label_column) 
    A_eq = build_A_eq_array(df, proxy_columns, label_column, v_arrays)
    b_eq = build_b(df, proxy_columns, protected_columns, include_simplex_constraints=True)
    for j in range(num_groups):
        flattened_r_array_j = get_flattened_v_array(r_array, j, num_groups)
        # flattened_r_array_sum = np.add(flattened_r_array_j)
        c = flattened_r_array_j
        res = linprog(c, A_eq = A_eq, b_eq=b_eq)
        optimal_value = res.fun
        optimal_W = res.x
        #print("optimal_value", optimal_value)
        W_term = optimal_value/(true_group_marginals[j] * float(len(df)))
        robust_constraint = tpr_overall - W_term - max_diff
        robust_constraints.append(robust_constraint)
    return robust_constraints

def get_robust_constraints_tpr(df, W, proxy_columns, label_column, true_group_marginals, max_diff=0.05):
    """Computes robust constraints for softweights using an exising W."""
    # Compute r
    r_array_1, r_array_2 = get_r_from_data_tpr(df, proxy_columns, label_column)

    robust_constraints = []
    tpr_overall = losses.tpr(df, label_column)
    for j in range(len(true_group_marginals)):
        Wterm_numerator = np.dot(r_array_1, W.T[j])
        Wterm_denominator = np.dot(r_array_2, W.T[j])
        Wterm = Wterm_numerator/Wterm_denominator
        Wterm = min(Wterm, 1) # W term should not be greater than 1.
        robust_constraint = tpr_overall - Wterm - max_diff
        robust_constraints.append(robust_constraint)
    return robust_constraints


def get_robust_constraints_fpr(df, W, proxy_columns, label_column, true_group_marginals, max_diff=0.05):
    """Computes robust constraints for softweights using an exising W."""
    # Compute r
    r_array_1, r_array_2 = get_r_from_data_fpr(df, proxy_columns, label_column)

    robust_constraints = []
    fpr_overall = losses.fpr(df, label_column)
    for j in range(len(true_group_marginals)):
        Wterm_numerator = np.dot(r_array_1, W.T[j])
        Wterm_denominator = np.dot(r_array_2, W.T[j])
        Wterm = Wterm_numerator/Wterm_denominator
        Wterm = min(Wterm, 1) # W term should not be greater than 1.
        robust_constraint =  Wterm - fpr_overall - max_diff
        robust_constraints.append(robust_constraint)
    return robust_constraints



def get_robust_constraints_err(df, W, proxy_columns, label_column, true_group_marginals, max_diff=0.05):
    """Computes robust constraints for softweights using an exising W."""
    # Compute r
    r_array = get_r_from_data_err(df, proxy_columns, label_column)

    # Robust equal error rates constraint.
    robust_constraints = []
    error_rate_overall = losses.error_rate(df[['predictions']], df[[label_column]])
    for j in range(len(true_group_marginals)):
        W_term = np.dot(r_array, W.T[j])/(true_group_marginals[j] * float(len(df)))
        W_term = min(W_term, 1) # W term should not be greater than 1.
        robust_constraint = W_term - error_rate_overall - max_diff
        robust_constraints.append(robust_constraint)
    return robust_constraints


def get_robust_constraints(df, W, proxy_columns, label_column, true_group_marginals, constraint='tpr', max_diff=0.05):
    """Computes robust constraints for softweights using an exising W."""
    # Compute r
    robust_constraints = None
    if constraint == 'err':
        robust_constraints = get_robust_constraints_err(df, W, proxy_columns, label_column, true_group_marginals, max_diff=max_diff)
    elif constraint == 'tpr':
        robust_constraints = get_robust_constraints_tpr(df, W, proxy_columns, label_column, true_group_marginals, max_diff=max_diff)
    elif constraint == 'fpr':
        robust_constraints = get_robust_constraints_fpr(df, W, proxy_columns, label_column, true_group_marginals, max_diff=max_diff)
    elif constraint == 'tpr_and_fpr':
        robust_constraints_tpr = get_robust_constraints_tpr(df, W, proxy_columns, label_column, true_group_marginals, max_diff=max_diff)
        robust_constraints_fpr = get_robust_constraints_fpr(df, W, proxy_columns, label_column, true_group_marginals, max_diff=max_diff)
        robust_constraints = robust_constraints_tpr + robust_constraints_fpr
    else:
        raise("constraint not supported.")
    return robust_constraints



def get_error_rate_and_constraints_softweights(df, W, protected_columns, proxy_columns, label_column, 
    true_group_marginals, constraint='tpr', max_diff=0.05, optimize_robust_constraints=False):
    """Computes the error and fairness violations. Currently only computes tpr violations.
    
    Args:
      df: dataframe containing 'predictions' column and LABEL_COLUMN, PROTECTED_COLUMNS, and PROXY_COLUMNS.
        predictions column is not required to be thresholded.
    
    """
    error_rate_overall, true_G_constraints, proxy_Ghat_constraints = losses.get_error_rate_and_constraints(df, protected_columns, proxy_columns, label_column, constraint=constraint, max_diff=max_diff)
    robust_constraints = None
    if optimize_robust_constraints:
        raise("optimized robust constraints not updated and not supported currently.")
        robust_constraints = get_optimized_robust_constraints(df, proxy_columns, protected_columns, label_column, true_group_marginals)
    else:
        robust_constraints = get_robust_constraints(df, W, proxy_columns, label_column, true_group_marginals, constraint=constraint, max_diff=max_diff)
    return error_rate_overall, true_G_constraints, proxy_Ghat_constraints, robust_constraints


def training_helper(sw_model,
                    train_df,
                    val_df,
                    test_df,
                    protected_columns, 
                    proxy_columns, 
                    label_column,
                    minibatch_size = None,
                    num_iterations_per_loop=1,
                    num_loops=1,
                    optimize_robust_constraints=False,
                    num_iterations_W=1,
                    max_diff=0.05,
                    constraint='tpr'):
    train_hinge_objective_vector = []
    # Hinge loss constraint violations on the proxy groups.
    train_hinge_constraints_matrix = []
    train_01_objective_vector = []
    train_01_true_G_constraints_matrix = []
    train_01_proxy_Ghat_constraints_matrix = []
    train_01_robust_constraints_matrix = []

    lambda_variables_matrix = []
    W_variables_matrix = []
    
    val_01_objective_vector = []
    val_01_true_G_constraints_matrix = []
    val_01_proxy_Ghat_constraints_matrix = []
    val_01_robust_constraints_matrix = []
    
    # List of T scalar values representing the 01 objective at each iteration.
    test_01_objective_vector = []
    # List of T vectors of size m, where each vector[i] is the zero-one constraint violation for group i.
    # Eventually we will just pick the last vector in this list, and take the max over m entries to get the max constraint violation.
    test_01_true_G_constraints_matrix = []
    test_01_proxy_Ghat_constraints_matrix = []
    test_01_robust_constraints_matrix = []

    true_group_marginals = get_true_group_marginals(train_df, protected_columns)
    for objective, constraints, train_predictions, lambda_variables, W_variable, val_predictions, test_predictions in training_generator(
      sw_model, train_df, val_df, test_df, minibatch_size, num_iterations_per_loop,
      num_loops, num_iterations_W=num_iterations_W):
        train_hinge_objective_vector.append(objective)
        train_hinge_constraints_matrix.append(constraints)
        
        train_df.loc[:, 'predictions'] = train_predictions
        train_01_objective, train_01_true_G_constraints, train_01_proxy_Ghat_constraints, train_01_robust_constraints = get_error_rate_and_constraints_softweights(
            train_df, W_variable, protected_columns, proxy_columns, label_column, true_group_marginals, optimize_robust_constraints=optimize_robust_constraints, max_diff=max_diff, constraint=constraint)
        train_01_objective_vector.append(train_01_objective)
        train_01_true_G_constraints_matrix.append(train_01_true_G_constraints)
        train_01_proxy_Ghat_constraints_matrix.append(train_01_proxy_Ghat_constraints)
        train_01_robust_constraints_matrix.append(train_01_robust_constraints)
        
        lambda_variables_matrix.append(lambda_variables)
        W_variables_matrix.append(W_variable)
        
        val_df.loc[:, 'predictions'] = val_predictions
        val_01_objective, val_01_true_G_constraints, val_01_proxy_Ghat_constraints, val_01_robust_constraints = get_error_rate_and_constraints_softweights(
            val_df, W_variable, protected_columns, proxy_columns, label_column, true_group_marginals, optimize_robust_constraints=optimize_robust_constraints, max_diff=max_diff, constraint=constraint)
        val_01_objective_vector.append(val_01_objective)
        val_01_true_G_constraints_matrix.append(val_01_true_G_constraints)
        val_01_proxy_Ghat_constraints_matrix.append(val_01_proxy_Ghat_constraints)
        val_01_robust_constraints_matrix.append(val_01_robust_constraints)
        
        test_df.loc[:, 'predictions'] = test_predictions
        test_01_objective, test_01_true_G_constraints, test_01_proxy_Ghat_constraints, test_01_robust_constraints = get_error_rate_and_constraints_softweights(
            test_df, W_variable, protected_columns, proxy_columns, label_column, true_group_marginals, optimize_robust_constraints=optimize_robust_constraints, max_diff=max_diff, constraint=constraint)
        test_01_objective_vector.append(test_01_objective)
        test_01_true_G_constraints_matrix.append(test_01_true_G_constraints)
        test_01_proxy_Ghat_constraints_matrix.append(test_01_proxy_Ghat_constraints)
        test_01_robust_constraints_matrix.append(test_01_robust_constraints)
        
    return {'train_hinge_objective_vector': train_hinge_objective_vector, 
            'train_hinge_constraints_matrix': train_hinge_constraints_matrix, 
            'train_01_objective_vector': train_01_objective_vector, 
            'train_01_true_G_constraints_matrix': train_01_true_G_constraints_matrix, 
            'train_01_proxy_Ghat_constraints_matrix': train_01_proxy_Ghat_constraints_matrix, 
            'train_01_robust_constraints_matrix': train_01_robust_constraints_matrix, 
            'lambda_variables_matrix': lambda_variables_matrix, 
            'W_variables_matrix': W_variables_matrix, 
            'val_01_objective_vector': val_01_objective_vector, 
            'val_01_true_G_constraints_matrix': val_01_true_G_constraints_matrix, 
            'val_01_proxy_Ghat_constraints_matrix': val_01_proxy_Ghat_constraints_matrix,
            'val_01_robust_constraints_matrix': val_01_robust_constraints_matrix,
            'test_01_objective_vector': test_01_objective_vector, 
            'test_01_true_G_constraints_matrix': test_01_true_G_constraints_matrix, 
            'test_01_proxy_Ghat_constraints_matrix': test_01_proxy_Ghat_constraints_matrix,
            'test_01_robust_constraints_matrix': test_01_robust_constraints_matrix}


def build_b(input_df, proxy_groups, true_groups, include_simplex_constraints=False):
    # If a proxy group has zero examples, appends 0.
    num_groups = len(proxy_groups)
    b = []
    for j in range(num_groups):
        for k in range(num_groups):
            # number of examples with proxy = k
            num_proxy = input_df[proxy_groups[k]].sum()
            if num_proxy == 0:
                b.append(0)
            else:
                # number of examples with true = j, proxy = k
                true_and_proxy = np.multiply(input_df[true_groups[j]],input_df[proxy_groups[k]])
                num_true_and_proxy = true_and_proxy.sum()
                # b_{jk} = P(G = j | \hat{G} = k)
                b.append(num_true_and_proxy / num_proxy)
    if include_simplex_constraints:
        W_num_rows = num_groups*4
        for i in range(W_num_rows):
            b.append(1)
    return np.array(b)


def get_true_group_marginals(input_df, true_groups):
    true_group_marginals = []
    for group in true_groups:
        marginal = input_df[group].mean()
        true_group_marginals.append(marginal)
    return true_group_marginals


def get_results_for_learning_rates(input_df, 
                                    feature_names, protected_columns, proxy_columns, label_column, 
                                    constraint = 'tpr', 
                                    learning_rates_theta = [0.1], 
                                    learning_rates_lambda = [1], 
                                    learning_rates_W = [0.1], 
                                    num_runs=10, 
                                    minibatch_size=None, 
                                    num_iterations_per_loop=25, 
                                    num_loops=30, 
                                    constraints_slack=0.0,
                                    num_avg_iters=0,
                                    optimize_robust_constraints=False,
                                    rank_objectives=False, # parameters for find_best_candidate_index
                                    max_constraints=False, # parameters for find_best_candidate_index
                                    num_iterations_W=5,
                                    max_diff=0.05,
                                    best_index_nburn=0, # Number of initial candidate indices to exclude from find_best_candidate_index.
                                    seed_start=100,
                                    ):    
    ts = time.time()
    # 10 runs with mean and stddev
    results_dicts_runs = []
    for i in range(num_runs):
        print('Split %d of %d' % (i, num_runs))
        t_split = time.time()

        train_df, val_df, test_df = data.train_val_test_split(input_df, 0.6, 0.2, seed=seed_start+i)
        # Refresh the b parameter and true_group_marginals parameter for every split.
        b = build_b(train_df, proxy_columns, protected_columns)
        true_group_marginals = get_true_group_marginals(train_df, protected_columns)

        val_objectives = []
        val_constraints_matrix = []
        results_dicts = []
        learning_rates_iters_theta = []
        learning_rates_iters_lambda = []
        learning_rates_iters_W = []

        for learning_rate_theta in learning_rates_theta:
            for learning_rate_lambda in learning_rates_lambda:
                for learning_rate_W in learning_rates_W:
                    t_start_iter = time.time() - ts
                    print("Time since start:", t_start_iter)
                    print("Starting optimizing learning rate theta: %.3f, learning rate lambda: %.3f, learning rate W: %.3f" % (learning_rate_theta, learning_rate_lambda, learning_rate_W))
                    sw_model = SoftweightsHeuristicModel(b, true_group_marginals, feature_names, proxy_columns, label_column, maximum_lambda_radius=1.0)
                    sw_model.build_train_ops(constraint=constraint, learning_rate_theta=learning_rate_theta, learning_rate_lambda=learning_rate_lambda, learning_rate_W=learning_rate_W, constraints_slack=constraints_slack)

                    # training_helper returns the list of errors and violations over each epoch.
                    results_dict = training_helper(
                          sw_model,
                          train_df,
                          val_df,
                          test_df,
                          protected_columns, 
                          proxy_columns, 
                          label_column,
                          minibatch_size=minibatch_size,
                          num_iterations_per_loop=num_iterations_per_loop,
                          num_loops=num_loops,
                          optimize_robust_constraints=optimize_robust_constraints,
                          num_iterations_W=num_iterations_W,
                          max_diff=max_diff,
                          constraint=constraint)
                    
                    # Get best iterate using training set.
                    best_index_iters = utils.find_best_candidate_index(np.array(results_dict['train_01_objective_vector'][best_index_nburn:]),np.array(results_dict['train_01_robust_constraints_matrix'][best_index_nburn:]), rank_objectives=rank_objectives, max_constraints=max_constraints)
                    best_index_iters = best_index_iters + best_index_nburn
                    results_dict_best_idx = add_results_dict_best_idx(results_dict, best_index_iters)
                    results_dicts.append(results_dict_best_idx)
                    if num_avg_iters == 0:
                        best_val_objective = results_dict['val_01_objective_vector'][best_index_iters]
                        best_val_constraints = results_dict['val_01_true_G_constraints_matrix'][best_index_iters]
                        val_objectives.append(best_val_objective)
                        val_constraints_matrix.append(best_val_constraints)
                        print ("best val objective: %0.4f" % best_val_objective)
                        print ("best val constraints:", best_val_constraints)
                    else: 
                        assert(num_avg_iters > 0)
                        avg_val_objective = np.mean(np.array(results_dict['val_01_objective_vector'][-num_avg_iters:]))
                        val_objectives.append(avg_val_objective)
                        avg_val_constraints = np.mean(np.array(results_dict['val_01_robust_constraints_matrix'][-num_avg_iters:]), axis=0)
                        val_constraints_matrix.append(avg_val_constraints)
                        print ("avg val objective: %0.4f" % avg_val_objective)
                        print ("avg val constraints:", avg_val_constraints)
                    learning_rates_iters_theta.append(learning_rate_theta)
                    learning_rates_iters_lambda.append(learning_rate_lambda)
                    learning_rates_iters_W.append(learning_rate_W)
                    print("Finished optimizing learning rate theta: %.3f, learning rate lambda: %.3f, learning rate W: %.3f" % (learning_rate_theta, learning_rate_lambda, learning_rate_W))
                    print("Time that this run took:", time.time() - t_start_iter - ts)
                
        # Get best hyperparameters using validation set.
        best_index = utils.find_best_candidate_index(np.array(val_objectives),np.array(val_constraints_matrix), rank_objectives=rank_objectives, max_constraints=max_constraints)
        best_results_dict = results_dicts[best_index]
        best_learning_rate_theta = learning_rates_iters_theta[best_index]
        best_learning_rate_lambda = learning_rates_iters_lambda[best_index]
        best_learning_rate_W = learning_rates_iters_W[best_index]
        print('best_learning_rate_theta,', best_learning_rate_theta)
        print('best_learning_rate_lambda', best_learning_rate_lambda)
        print('best_learning_rate_W', best_learning_rate_W)
        results_dicts_runs.append(best_results_dict)
        print("time it took for this split", time.time() - t_split)
        print('best true G constraint violations', best_results_dict['best_train_01_true_G_constraints_matrix'])
    final_average_results_dict = utils.average_results_dict_fn(results_dicts_runs)
    
    return final_average_results_dict


def add_results_dict_best_idx(results_dict, best_index):
    columns_to_add = ['train_01_objective_vector', 'train_01_true_G_constraints_matrix', 'train_01_proxy_Ghat_constraints_matrix', 'train_01_robust_constraints_matrix', 
                     'val_01_objective_vector', 'val_01_true_G_constraints_matrix', 'val_01_proxy_Ghat_constraints_matrix', 'val_01_robust_constraints_matrix', 
                     'test_01_objective_vector', 'test_01_true_G_constraints_matrix', 'test_01_proxy_Ghat_constraints_matrix', 'test_01_robust_constraints_matrix']
    for column in columns_to_add:
        results_dict['best_' + column] = results_dict[column][best_index]
    return results_dict


def train_one_model(input_df, 
                    feature_names, protected_columns, proxy_columns, label_column, 
                    constraint = 'tpr', 
                    learning_rate_theta = 0.01, 
                    learning_rate_lambda = 1, 
                    learning_rate_W = 0.01,
                    minibatch_size=None, 
                    num_iterations_per_loop=25, 
                    num_loops=30, 
                    constraints_slack=0.0,
                    num_avg_iters=10,
                    rank_objectives=False, # parameters for find_best_candidate_index
                    max_constraints=False, # parameters for find_best_candidate_index
                    num_iterations_W=1,
                    best_index_nburn=0
                    ):    
    train_df, val_df, test_df = data.train_val_test_split(input_df, 0.6, 0.2, seed=88)
    b = build_b(train_df, proxy_columns, protected_columns)
    true_group_marginals = get_true_group_marginals(train_df, protected_columns)
    
    sw_model = SoftweightsHeuristicModel(b, true_group_marginals, feature_names, proxy_columns, label_column, maximum_lambda_radius=2)
    sw_model.build_train_ops(constraint=constraint, learning_rate_theta=learning_rate_theta, learning_rate_lambda=learning_rate_lambda, learning_rate_W=learning_rate_W, constraints_slack=constraints_slack)

    # training_helper returns the list of errors and violations over each epoch.
    results_dict = training_helper(
          sw_model,
          train_df,
          val_df,
          test_df,
          protected_columns, 
          proxy_columns, 
          label_column,
          minibatch_size=minibatch_size,
          num_iterations_per_loop=num_iterations_per_loop,
          num_loops=num_loops,
          num_iterations_W=num_iterations_W)

    best_index_iters = utils.find_best_candidate_index(np.array(results_dict['train_01_objective_vector'][best_index_nburn:]),np.array(results_dict['train_01_robust_constraints_matrix'][best_index_nburn:]), rank_objectives=rank_objectives, max_constraints=max_constraints)
    best_index_iters = best_index_iters + best_index_nburn
    results_dict_best_idx = add_results_dict_best_idx(results_dict, best_index_iters)
    return results_dict_best_idx, (train_df, val_df, test_df)


# Expects results dicts without averaging.
def print_results_last_iter(results_dict, iter_num = -1):
    print("last iterate 0/1 objectives: (train, val, test)")
    print("%.4f, %.4f ,%.4f " % (results_dict['train_01_objective_vector'][iter_num], 
                                 results_dict['val_01_objective_vector'][iter_num],
                                 results_dict['test_01_objective_vector'][iter_num]))

    print("last iterate 0/1 true G constraints: (train, val, test)")
    print("%.4f, %.4f ,%.4f" % (max(results_dict['train_01_true_G_constraints_matrix'][iter_num]), 
                                max(results_dict['val_01_true_G_constraints_matrix'][iter_num]),
                                max(results_dict['test_01_true_G_constraints_matrix'][iter_num])))
    
    print("last iterate 0/1 proxy Ghat constraints: (train, val, test)")
    print("%.4f, %.4f ,%.4f" % (max(results_dict['train_01_proxy_Ghat_constraints_matrix'][iter_num]), 
                                max(results_dict['val_01_proxy_Ghat_constraints_matrix'][iter_num]),
                                max(results_dict['test_01_proxy_Ghat_constraints_matrix'][iter_num])))

    print("last iterate 0/1 robust constraints: (train, val, test)")
    print("%.4f, %.4f ,%.4f" % (max(results_dict['train_01_robust_constraints_matrix'][iter_num]), 
                                max(results_dict['val_01_robust_constraints_matrix'][iter_num]),
                                max(results_dict['test_01_robust_constraints_matrix'][iter_num])))

# Expects results dicts without averaging.
def print_results_best_iter(results_dict):
    print("best iterate 0/1 objectives: (train, val, test)")
    print("%.4f, %.4f ,%.4f " % (results_dict['best_train_01_objective_vector'], 
                                 results_dict['best_val_01_objective_vector'],
                                 results_dict['best_test_01_objective_vector']))

    print("best iterate 0/1 true G constraints: (train, val, test)")
    print("%.4f, %.4f ,%.4f" % (max(results_dict['best_train_01_true_G_constraints_matrix']), 
                                max(results_dict['best_val_01_true_G_constraints_matrix']),
                                max(results_dict['best_test_01_true_G_constraints_matrix'])))
    
    print("best iterate 0/1 proxy Ghat constraints: (train, val, test)")
    print("%.4f, %.4f ,%.4f" % (max(results_dict['best_train_01_proxy_Ghat_constraints_matrix']), 
                                max(results_dict['best_val_01_proxy_Ghat_constraints_matrix']),
                                max(results_dict['best_test_01_proxy_Ghat_constraints_matrix'])))

    print("best iterate 0/1 robust constraints: (train, val, test)")
    print("%.4f, %.4f ,%.4f" % (max(results_dict['best_train_01_robust_constraints_matrix']), 
                                max(results_dict['best_val_01_robust_constraints_matrix']),
                                max(results_dict['best_test_01_robust_constraints_matrix'])))


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

    print("avg iterate 0/1 robust constraints: (train, val, test)")
    print("%.4f, %.4f ,%.4f" % (np.max(np.mean(np.array(results_dict['train_01_robust_constraints_matrix'][-num_avg_iters:]), axis=0)), 
                                np.max(np.mean(np.array(results_dict['val_01_robust_constraints_matrix'][-num_avg_iters:]), axis=0)),
                                np.max(np.mean(np.array(results_dict['test_01_robust_constraints_matrix'][-num_avg_iters:]), axis=0))))


# Expects results dicts without averaging.
def plot_optimization_softweights(results_dict):
    fig, axs = plt.subplots(5, figsize=(5,25))
    axs[0].plot(results_dict['train_hinge_objective_vector'])
    axs[0].set_title('train_hinge_objective_vector')
    axs[1].plot(results_dict['train_hinge_constraints_matrix'])
    axs[1].set_title('train_hinge_constraints_matrix')
    axs[2].plot(results_dict['train_01_proxy_Ghat_constraints_matrix'])
    axs[2].set_title('train_01_proxy_Ghat_constraints_matrix')
    axs[3].plot(results_dict['train_01_true_G_constraints_matrix'])
    axs[3].set_title('train_01_true_G_constraints_matrix')
    axs[4].plot(results_dict['train_01_robust_constraints_matrix'])
    axs[4].set_title('train_01_robust_constraints_matrix')
    plt.show()


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

    print("best iterate max 0/1 robust constraint violations: (train, val, test)")
    train_max_mean_std = get_max_mean_std_best(results_dict['best_train_01_robust_constraints_matrix'])
    val_max_mean_std = get_max_mean_std_best(results_dict['best_val_01_robust_constraints_matrix'])
    test_max_mean_std = get_max_mean_std_best(results_dict['best_test_01_robust_constraints_matrix'])
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

    print("avg iterate 0/1 robust constraints: (train, val, test)")
    train_max_mean_std = get_max_mean_std(results_dict['train_01_robust_constraints_matrix'])
    val_max_mean_std = get_max_mean_std(results_dict['val_01_robust_constraints_matrix'])
    test_max_mean_std = get_max_mean_std(results_dict['test_01_robust_constraints_matrix'])
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_std[0], train_max_mean_std[1], 
                                                     val_max_mean_std[0], val_max_mean_std[1],
                                                     test_max_mean_std[0], test_max_mean_std[1]))



# Expects averaged results_dict with means and standard deviations.
def print_avg_results_last_iter(results_dict, iter_num=-1):
    def get_max_mean_std(mean_std_tuple):
        max_idx = np.argmax(mean_std_tuple[0][iter_num])
        max_mean = mean_std_tuple[0][iter_num][max_idx]
        max_std = mean_std_tuple[1][iter_num][max_idx]
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

    print("last iterate 0/1 robust constraints: (train, val, test)")
    train_max_mean_std = get_max_mean_std(results_dict['train_01_robust_constraints_matrix'])
    val_max_mean_std = get_max_mean_std(results_dict['val_01_robust_constraints_matrix'])
    test_max_mean_std = get_max_mean_std(results_dict['test_01_robust_constraints_matrix'])
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

    train_max_mean_tpr, train_max_std_tpr, train_max_mean_fpr, train_max_std_fpr = get_max_mean_std_best(results_dict['best_train_01_robust_constraints_matrix'])
    val_max_mean_tpr, val_max_std_tpr, val_max_mean_fpr, val_max_std_fpr = get_max_mean_std_best(results_dict['best_train_01_robust_constraints_matrix'])
    test_max_mean_tpr, test_max_std_tpr, test_max_mean_fpr, test_max_std_fpr = get_max_mean_std_best(results_dict['best_train_01_robust_constraints_matrix'])

    print("best iterate max 0/1 robust constraint violations (TPR): (train, val, test)")
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_tpr, train_max_std_tpr, 
                                                         val_max_mean_tpr, val_max_std_tpr,
                                                         test_max_mean_tpr, test_max_std_tpr))   

    print("best iterate max 0/1 robust constraint violations (FPR): (train, val, test)")
    print("%.4f \pm %.4f,%.4f \pm %.4f,%.4f \pm %.4f" % (train_max_mean_fpr, train_max_std_fpr, 
                                                         val_max_mean_fpr, val_max_std_fpr,
                                                         test_max_mean_fpr, test_max_std_fpr))
       


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
        axs[4].errorbar(iters, results_dict['train_01_robust_constraints_matrix'][0].T[i], results_dict['train_01_robust_constraints_matrix'][1].T[i], label=protected_columns[i])
    axs[1].set_title('train_hinge_constraints_matrix')
    axs[1].legend()
    axs[2].set_title('train_01_proxy_Ghat_constraints_matrix')
    axs[2].legend()
    axs[3].set_title('train_01_true_G_constraints_matrix')
    axs[3].legend()
    axs[4].set_title('train_01_robust_constraints_matrix')
    axs[4].legend()
    plt.show()

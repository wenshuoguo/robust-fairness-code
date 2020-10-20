"""General Model class for constrained optimization."""

import numpy as np
import tensorflow as tf

import losses
import optimization
import utils


class Model(object):
    """Linear model for performing constrained optimization.
    
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
    def __init__(self, feature_names, protected_columns, label_column, maximum_lambda_radius=None, hidden_layer_size=0):
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.random.set_random_seed(123)
        
        self.feature_names = feature_names
        self.protected_columns = protected_columns
        self.label_column = label_column

        if (maximum_lambda_radius is not None and maximum_lambda_radius <= 0.0):
            raise ValueError("maximum_lambda_radius must be strictly positive")
        self._maximum_lambda_radius = maximum_lambda_radius
        
        # Set up feature and label tensors.
        num_features = len(self.feature_names)
        self.features_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, num_features), name='features_placeholder')
        self.protected_placeholders = [tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name=attribute+"_placeholder") for attribute in self.protected_columns]
        self.labels_placeholder = tf.compat.v1.placeholder(
            tf.float32, shape=(None, 1), name='labels_placeholder')
        self.num_groups = len(self.protected_placeholders)
        
        # Construct model.
        if hidden_layer_size > 0:
            hidden = tf.layers.dense(inputs=self.features_placeholder, units=hidden_layer_size, activation=tf.nn.relu,name="model0")
            self.predictions_tensor = tf.layers.dense(inputs=hidden, units=1, activation=None, name="model")
        else:
            self.predictions_tensor = tf.layers.dense(inputs=self.features_placeholder, units=1, activation=None, name="model")
        self.theta_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "model*")
        
    def feed_dict_helper(self, dataframe):
        feed_dict = {self.features_placeholder: dataframe[self.feature_names], self.labels_placeholder: dataframe[[self.label_column]],}
        for i, protected_attribute in enumerate(self.protected_columns):
                feed_dict[self.protected_placeholders[i]] = dataframe[[protected_attribute]]
        return feed_dict
    
    def project_lambdas(self, lambdas):
        """Projects the Lagrange multipliers onto the feasible region."""
        if self._maximum_lambda_radius:
            projected_lambdas = optimization.project_multipliers_wrt_euclidean_norm(
              lambdas, self._maximum_lambda_radius)
        else:
            projected_lambdas = tf.maximum(0.0, lambdas)
        return projected_lambdas
    
    def project_ptilde(self, ptilde, idx):
        """Projects the Lagrange multipliers ptildes onto the feasible region."""
        #print('phats shape', phats.shape)
        #phat = tf.gather(phats, [idx])
        #phat = tf.reshape(phat, (NUM_DATA,))
        phat = self.phats[idx, :]
        phat = tf.convert_to_tensor(phat)
        projected_ptilde = optimization.project_multipliers_to_L1_ball(ptilde, phat, self._maximum_p_radius[idx])
        return projected_ptilde
 
    def get_equal_fpr_constraints(self, constraints_slack=1.0):
        average_fpr = losses.concave_hinge_loss(self.labels_placeholder, self.predictions_tensor, weights=utils.flip_binary_tensor(self.labels_placeholder), reduction=tf.compat.v1.losses.Reduction.MEAN)
        constraints_list = [(tf.losses.hinge_loss(self.labels_placeholder, self.predictions_tensor, weights=tf.multiply(protected_placeholder, utils.flip_binary_tensor(self.labels_placeholder)), reduction=tf.compat.v1.losses.Reduction.MEAN) - average_fpr 
                            - (constraints_slack * tf.ones_like(average_fpr))) for protected_placeholder in self.protected_placeholders]
        return constraints_list

    def get_equal_fpr_constraints_nonconvex(self, constraints_slack=0.0):
        average_fpr = tf.losses.hinge_loss(self.labels_placeholder, self.predictions_tensor, weights=utils.flip_binary_tensor(self.labels_placeholder), reduction=tf.compat.v1.losses.Reduction.MEAN)
        constraints_list = [(tf.losses.hinge_loss(self.labels_placeholder, self.predictions_tensor, weights=tf.multiply(protected_placeholder, utils.flip_binary_tensor(self.labels_placeholder)), reduction=tf.compat.v1.losses.Reduction.MEAN) - average_fpr 
                            - (constraints_slack * tf.ones_like(average_fpr))) for protected_placeholder in self.protected_placeholders]
        return constraints_list

    def get_equal_tpr_constraints(self, constraints_slack=0.0):
        average_tpr = tf.losses.hinge_loss(utils.flip_binary_tensor(self.labels_placeholder), self.predictions_tensor, weights=self.labels_placeholder, reduction=tf.compat.v1.losses.Reduction.MEAN)
        constraints_list = [(average_tpr - losses.concave_hinge_loss(utils.flip_binary_tensor(self.labels_placeholder), self.predictions_tensor, weights=tf.multiply(protected_placeholder, self.labels_placeholder), reduction=tf.compat.v1.losses.Reduction.MEAN) 
                            - (constraints_slack * tf.ones_like(average_tpr))) for protected_placeholder in self.protected_placeholders]
        return constraints_list

    def get_equal_tpr_constraints_nonconvex(self, constraints_slack=0.0):
        average_tpr = tf.losses.hinge_loss(utils.flip_binary_tensor(self.labels_placeholder), self.predictions_tensor, weights=self.labels_placeholder, reduction=tf.compat.v1.losses.Reduction.MEAN)
        constraints_list = [(average_tpr - tf.losses.hinge_loss(utils.flip_binary_tensor(self.labels_placeholder), self.predictions_tensor, weights=tf.multiply(protected_placeholder, self.labels_placeholder), reduction=tf.compat.v1.losses.Reduction.MEAN) 
                            - (constraints_slack * tf.ones_like(average_tpr))) for protected_placeholder in self.protected_placeholders]
        return constraints_list

    def get_equal_tpr_and_fpr_constraints(self, constraints_slack=0.0):
        constraints_list_tpr = self.get_equal_tpr_constraints(constraints_slack=constraints_slack)
        constraints_list_fpr = self.get_equal_fpr_constraints(constraints_slack=constraints_slack)
        constraints_list = constraints_list_tpr + constraints_list_fpr
        return constraints_list
    
    def get_equal_accuracy_constraints(self, constraints_slack=1.0):
        average_concave_hinge = losses.concave_hinge_loss(self.labels_placeholder, self.predictions_tensor)
        constraints_list = [(tf.losses.hinge_loss(self.labels_placeholder, self.predictions_tensor, weights=protected_placeholder) - average_concave_hinge 
                            - (constraints_slack * tf.ones_like(average_concave_hinge))) for protected_placeholder in self.protected_placeholders]
        return constraints_list
     
    def get_equal_tpr_constraints_dro(self, constraints_slack=0.0):
        
        average_tpr = tf.losses.hinge_loss(utils.flip_binary_tensor(self.labels_placeholder), self.predictions_tensor, weights=self.labels_placeholder, reduction=tf.compat.v1.losses.Reduction.MEAN)
        constraints_list = [(average_tpr - losses.concave_hinge_loss(utils.flip_binary_tensor(
            self.labels_placeholder), self.predictions_tensor, 
        weights=tf.multiply(self.labels_placeholder, tf.reshape(p_variable,(self.num_data,1))), 
        reduction=tf.compat.v1.losses.Reduction.MEAN) - (constraints_slack * tf.ones_like(average_tpr))) for p_variable in self.p_variables_list]
        return constraints_list
    
    def get_equal_tpr_and_fpr_constraints_dro(self, constraints_slack=0.0):

            average_tpr = tf.losses.hinge_loss(utils.flip_binary_tensor(self.labels_placeholder), self.predictions_tensor, weights=self.labels_placeholder, reduction=tf.compat.v1.losses.Reduction.MEAN)
            
            constraints_list_tpr = [(average_tpr - losses.concave_hinge_loss(utils.flip_binary_tensor(
                self.labels_placeholder), self.predictions_tensor, 
            weights=tf.multiply(self.labels_placeholder, tf.reshape(p_variable,(self.num_data,1))), 
            reduction=tf.compat.v1.losses.Reduction.MEAN) - (constraints_slack * tf.ones_like(average_tpr))) for p_variable in self.p_variables_list]
            
            average_fpr = losses.concave_hinge_loss(self.labels_placeholder, self.predictions_tensor, weights=utils.flip_binary_tensor(self.labels_placeholder), reduction=tf.compat.v1.losses.Reduction.MEAN)
        
            constraints_list_fpr = [(tf.losses.hinge_loss(self.labels_placeholder, self.predictions_tensor, weights=tf.multiply(utils.flip_binary_tensor(self.labels_placeholder), tf.reshape(p_variable,(self.num_data,1))), reduction=tf.compat.v1.losses.Reduction.MEAN) - average_fpr - (constraints_slack * tf.ones_like(average_fpr))) for p_variable in self.p_variables_list]
            
            constraints_list = constraints_list_tpr + constraints_list_fpr
            
            return constraints_list
    
       
    def get_equal_accuracy_constraints_dro(self, constraints_slack=1.0):
        average_concave_hinge = losses.concave_hinge_loss(self.labels_placeholder, self.predictions_tensor)
        constraints_list =[(tf.losses.hinge_loss(self.labels_placeholder, self.predictions_tensor, weights=protected_placeholder, reduction=Reduction.SUM) - average_concave_hinge 
                            - (constraints_slack * tf.ones_like(average_concave_hinge))) for protected_placeholder in self.protected_placeholders]
        return constraints_list
   

       


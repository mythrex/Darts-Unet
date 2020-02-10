import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1.keras import Model

import numpy as np
import utils

def _concat(xs):
    """n-d tensor to 1d tensor

    Args:
        xs (array): the array of nd tensor

    Returns:
        array: concated array
    """
    return tf.concat([tf.reshape(x, [tf.size(x)]) for x in xs], axis=0, name="_concat")


class Architect(object):
    """Constructs the model

    Parameters:
      network_momentum(float):  network momentum
      network_weight_decay(float): network weight decay
      model(Network): Network archtecture with cells
      optimise(optimiser): Adam / SGD
    """

    def __init__(self, model, args):
        """Initialises the architecture

        Args:
            model (Network): Network archtecture with cells
            args (dict): cli args
        """
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.use_tpu = args.use_tpu
        
        self.arch_learning_rate = args.arch_learning_rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.arch_learning_rate,
                                                beta1=0.5,
                                                beta2=0.999)
        self.use_bfloat = args.use_bfloat
        if(self.use_tpu):
            self.optimizer = tf.tpu.CrossShardOptimizer(self.optimizer)
        
        self.learning_rate = args.learning_rate

    def get_model_theta(self, model):
        specific_tensor = []
        specific_tensor_name = []
        for var in model.trainable_weights:
            if not 'alphas' in var.name:
                specific_tensor.append(var)
                specific_tensor_name.append(var.name)
        return specific_tensor
    
    def step(self, input_train, target_train, input_valid, target_valid, unrolled):
        """Computer a step for gradient descend

        Args:
            input_train (tensor): a train of input
            target_train (tensor): a train of targets
            input_valid (tensor): a train of validation
            target_valid (tensor): a train of validation targets
            eta (tensor): eta
            network_optimizer (optimiser): network optimiser for network
            unrolled (bool): True if training we need unrolled
        """
        if unrolled:
            logits = self.model(input_train)
            train_loss = self.model._loss(logits, target_train)
            return self._compute_unrolled_step(input_train, 
                                               target_train, 
                                               input_valid, 
                                               target_valid,
                                               self.get_model_theta(self.model),
                                               train_loss,
                                               self.learning_rate
                                              )
        else:
            return self._backward_step(input_valid, target_valid)
        
    
    def _compute_unrolled_step(self, x_train, y_train, x_valid, y_valid, w_var, train_loss, lr):
        arch_var = self.model.arch_parameters()
        
        unrolled_model = self.model.new()
        _ = unrolled_model(x_train)
        unrolled_w_var = self.get_model_theta(unrolled_model)
        copy_weight_opts = [v.assign(w) for v,w in zip(unrolled_w_var, w_var)]
        logits = unrolled_model(x_train)
        
        unrolled_train_loss = unrolled_model._loss(logits, y_train)  

        with tf.control_dependencies(copy_weight_opts):
            unrolled_optimizer = tf.train.GradientDescentOptimizer(lr)
            if(self.use_tpu):
                unrolled_optimizer = tf.tpu.CrossShardOptimizer(unrolled_optimizer)
            update_ops1 = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops1):
                unrolled_optimizer = unrolled_optimizer.minimize(unrolled_train_loss, var_list=unrolled_w_var)

        valid_logits = unrolled_model(x_valid)
        valid_loss = unrolled_model._loss(valid_logits, y_valid)

        with tf.control_dependencies([unrolled_optimizer]):
            valid_grads = self.optimizer.compute_gradients(valid_loss, var_list=unrolled_w_var)
            valid_grads = list(map(lambda x: x[0], valid_grads))
            
        r=1e-2
        smoothy = 1e-6
        if(self.use_bfloat):
            R = 1e-10
        else:
            R = r / (tf.linalg.global_norm(valid_grads)+ smoothy)

        optimizer_pos=tf.train.GradientDescentOptimizer(R)
        if(self.use_tpu):
            optimizer_pos = tf.tpu.CrossShardOptimizer(optimizer_pos)
        update_ops2 = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops2):
            optimizer_pos_op=optimizer_pos.apply_gradients(zip(valid_grads, w_var))

        optimizer_neg=tf.train.GradientDescentOptimizer(-2*R)
        if(self.use_tpu):
            optimizer_neg = tf.tpu.CrossShardOptimizer(optimizer_neg)
        update_ops3 = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops3):
            optimizer_neg_op=optimizer_neg.apply_gradients(zip(valid_grads, w_var))

        optimizer_back=tf.train.GradientDescentOptimizer(R)
        if(self.use_tpu):
            optimizer_back = tf.tpu.CrossShardOptimizer(optimizer_back)
        update_ops4 = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops4):
            optimizer_back_op=optimizer_back.apply_gradients(zip(valid_grads, w_var))
        
        with tf.control_dependencies([optimizer_pos_op]):
            train_grads_pos= optimizer_pos.compute_gradients(train_loss, var_list=arch_var)
            train_grads_pos = list(map(lambda x: x[0], train_grads_pos))
            print(train_grads_pos)
            with tf.control_dependencies([optimizer_neg_op]):
                train_grads_neg=optimizer_neg.compute_gradients(train_loss,  var_list=arch_var)
                train_grads_neg = list(map(lambda x: x[0], train_grads_neg))
                with tf.control_dependencies([optimizer_back_op]):
                    leader_opt= self.optimizer
                    leader_grads=self.optimizer.compute_gradients(valid_loss, var_list=unrolled_model.arch_parameters())
                    leader_grads = list(map(lambda x: x[0], leader_grads))
        
        for i,g in enumerate(leader_grads):
            leader_grads[i] = g - self.learning_rate * tf.divide(train_grads_pos[i]-train_grads_neg[i],2*R)
        update_ops5 = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops5):
            leader_opt=leader_opt.apply_gradients(zip(leader_grads, arch_var))
        return leader_opt
    
    def _backward_step(self, input_valid, target_valid):
        """Backward step for validation

        Args:
            input_train (tensor): a train of input
            target_train (tensor): a train of targets
        """
        loss = self.model._loss(self.model(input_valid), target_valid)
        update_ops1 = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops1):
            opt = self.optimizer.minimize(loss, var_list=model.get_weights())
        return opt
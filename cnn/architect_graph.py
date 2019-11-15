import tensorflow as tf
from tensorflow.keras import Model
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
        self.arch_learning_rate = args.arch_learning_rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.arch_learning_rate,
                                                beta1=0.5,
                                                beta2=0.999)  
        global_step = tf.train.get_global_step()
        learning_rate_min = tf.constant(args.learning_rate_min)

        learning_rate = tf.train.exponential_decay(
            args.learning_rate,
            global_step,
            decay_rate=args.learning_rate_decay,
            decay_steps=args.num_batches_per_epoch,
            staircase=True,
        )

        lr = tf.maximum(learning_rate, learning_rate_min)
        
        self.learning_rate = lr

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
            w_regularization_loss = 0.25
            logits = self.model(input_train)
            train_loss = self.model._loss(logits, target_train)
            train_loss += 1e4*0.25*w_regularization_loss
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
        logits = unrolled_model(x_train)
        unrolled_train_loss = unrolled_model._loss(logits, y_train)  
        unrolled_w_var = self.get_model_theta(unrolled_model)
        copy_weight_opts = [v.assign(w) for v,w in zip(unrolled_w_var,w_var)]
        #w'
        with tf.control_dependencies(copy_weight_opts):
            unrolled_optimizer = tf.train.GradientDescentOptimizer(lr)
            unrolled_optimizer = unrolled_optimizer.minimize(unrolled_train_loss, var_list=unrolled_w_var)

        valid_logits = unrolled_model(x_valid)
        valid_loss = unrolled_model._loss(valid_logits, y_valid)
        tf.summary.scalar('valid_loss', valid_loss)

        with tf.control_dependencies([unrolled_optimizer]):
            valid_grads = tf.gradients(valid_loss, unrolled_w_var)

        r=1e-2
        R = r / (tf.global_norm(valid_grads)+1e-6)

        optimizer_pos=tf.train.GradientDescentOptimizer(R)
        optimizer_pos=optimizer_pos.apply_gradients(zip(valid_grads, w_var))

        optimizer_neg=tf.train.GradientDescentOptimizer(-2*R)
        optimizer_neg=optimizer_neg.apply_gradients(zip(valid_grads, w_var))

        optimizer_back=tf.train.GradientDescentOptimizer(R)
        optimizer_back=optimizer_back.apply_gradients(zip(valid_grads, w_var))

        with tf.control_dependencies([optimizer_pos]):
            train_grads_pos=tf.gradients(train_loss, arch_var)
            with tf.control_dependencies([optimizer_neg]):
                train_grads_neg=tf.gradients(train_loss,arch_var)	
                with tf.control_dependencies([optimizer_back]):
                  leader_opt= self.optimizer
                  leader_grads=leader_opt.compute_gradients(valid_loss, var_list =unrolled_model.arch_parameters())
        for i,(g,v) in enumerate(leader_grads):
            leader_grads[i]=(g - self.learning_rate * tf.divide(train_grads_pos[i]-train_grads_neg[i],2*R),v)

        leader_opt=leader_opt.apply_gradients(leader_grads)
        return leader_opt
    
    def _backward_step(self, input_valid, target_valid):
        """Backward step for validation

        Args:
            input_train (tensor): a train of input
            target_train (tensor): a train of targets
        """
        loss = self.model._loss(self.model(input_valid), target_valid)
        opt = self.optimizer.minimize(loss, var_list=model.get_weights())
        return opt
import tensorflow as tf
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
        self.learning_rate = args.learning_rate

    def get_model_theta(self, model):
        specific_tensor = []
        specific_tensor_name = []
        for var in model.get_weights():
            if not 'alphas' in var.name:
                specific_tensor.append(var)
                specific_tensor_name.append(var.name)
        return specific_tensor_name, specific_tensor

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
            self._compute_unrolled_step(
                input_train,
                target_train,
                input_valid,
                target_valid,
                self.get_model_theta(self.model)[1],
                self.learning_rate
            )
        else:
            self._backward_step(input_valid, target_valid)

    def _compute_unrolled_step(self, x_train, y_train, x_valid, y_valid, w_var, lr):
        arch_var = self.model.arch_parameters()
        unrolled_model = self.model.new()
        unrolled_optimizer = tf.train.GradientDescentOptimizer(lr)

        with tf.GradientTape() as tape:
            logits = unrolled_model(x_train)
            unrolled_w_var = self.get_model_theta(unrolled_model)[1]
            # copy weights
            for v, w in zip(unrolled_w_var, w_var):
                v.assign(w)
            unrolled_train_loss = unrolled_model._criterion(logits, y_train)
            grads = tape.gradient(unrolled_train_loss, unrolled_w_var)
            unrolled_optimizer.apply_gradients(zip(grads, unrolled_w_var))

        with tf.GradientTape() as tape1:
            valid_loss = unrolled_model._criterion(
                unrolled_model(x_valid), y_valid)
            valid_grads = tape1.gradient(valid_loss, unrolled_w_var)

        r = 1e-2
        R = r / (tf.global_norm(valid_grads)+1e-6)

        optimizer_pos = tf.train.GradientDescentOptimizer(R)
        optimizer_pos = optimizer_pos.apply_gradients(zip(valid_grads, w_var))

        optimizer_neg = tf.train.GradientDescentOptimizer(-2*R)
        optimizer_neg = optimizer_neg.apply_gradients(zip(valid_grads, w_var))

        optimizer_back = tf.train.GradientDescentOptimizer(R)
        optimizer_back = optimizer_back.apply_gradients(
            zip(valid_grads, w_var))

        with tf.GradientTape() as tape2:
            logits_model = self.model(x_train)
            train_loss = self.model._criterion(logits_model, y_train)
            train_grads_pos = tape2.gradient(train_loss, arch_var)

        with tf.GradientTape() as tape3:
            logits_model = self.model(x_train)
            train_loss = self.model._criterion(logits_model, y_train)
            train_grads_neg = tape3.gradient(train_loss, arch_var)

        with tf.GradientTape() as tape4:
            valid_loss = unrolled_model._criterion(
                unrolled_model(x_valid), y_valid)
            leader_grads = tape4.gradient(
                valid_loss, unrolled_model.arch_parameters())
        for i, (g, v) in enumerate(zip(leader_grads, arch_var)):
            leader_grads[i] = (
                g-lr*tf.divide(train_grads_pos[i]-train_grads_neg[i], 2*R), v)
        self.optimizer.apply_gradients(leader_grads)

    def _backward_step(self, input_valid, target_valid):
        """Backward step for validation

        Args:
            input_train (tensor): a train of input
            target_train (tensor): a train of targets
        """
        loss = self.model._loss(self.model(input_valid), target_valid)
        self.optimizer.minimize(loss, var_list=model.get_weights())

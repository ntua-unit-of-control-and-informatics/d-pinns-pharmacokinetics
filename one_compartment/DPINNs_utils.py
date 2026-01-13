# custom_pinn_utils.py

import numpy as np
import tensorflow as tf
from deepxde import optimizers, config
from deepxde.nn import activations, initializers, regularizers
from deepxde.display import TrainingDisplay
from deepxde import Model, config, optimizers
from deepxde.nn import regularizers, activations, initializers
from deepxde.nn.tensorflow.fnn import FNN
from deepxde.nn.tensorflow.nn import NN
import deepxde
import os


class PositiveOutputFNN(deepxde.nn.tensorflow.fnn.FNN):
    def __init__(
        self, layer_sizes, activation, kernel_initializer, regularization=("l2", 0.0)
    ):
        super().__init__(
            layer_sizes=layer_sizes,
            activation=activation,
            kernel_initializer=kernel_initializer,
            regularization=regularization,
        )

    def call(self, inputs, training=False):
        x = super().call(inputs, training=training)
        return tf.math.softplus(x) + 1e-10


def custom_call(self, train_state):
    if not self.is_header_print:
        self.len_train = len(train_state.loss_train) * 2 + 1
        self.len_test = len(train_state.loss_test) * 2 + 1
        self.len_metric = len(train_state.metrics_test) * 2 + 1
        self.header()
    self.print_one(
        str(train_state.step),
        np.sum(train_state.loss_train),
        np.sum(train_state.loss_test),
        train_state.metrics_test,
    )


def patch_training_display():
    TrainingDisplay.__call__ = custom_call


def custom_compile_tensorflow(self, lr, loss_fn, decay):
    """tensorflow"""

    @tf.function(jit_compile=config.xla_jit)
    def outputs(training, inputs):
        return self.net(inputs, training=training)

    def outputs_losses(training, inputs, targets, auxiliary_vars, losses_fn):
        self.net.auxiliary_vars = auxiliary_vars
        # Don't call outputs() decorated by @tf.function above, otherwise the
        # gradient of outputs wrt inputs will be lost here.
        outputs_ = self.net(inputs, training=training)
        # Data losses
        # if forward-mode AD is used, then a forward call needs to be passed
        aux = [self.net] if config.autodiff == "forward" else None
        losses = losses_fn(targets, outputs_, loss_fn, inputs, self, aux=aux)
        if not isinstance(losses, list):
            losses = [losses]
        # Regularization loss
        if self.net.regularizer is not None:
            losses += [tf.math.reduce_sum(self.net.losses)]
        losses = tf.convert_to_tensor(losses)
        # Weighted losses
        if self.loss_weights is not None:
            losses *= self.loss_weights
        return outputs_, losses

    @tf.function(jit_compile=config.xla_jit)
    def outputs_losses_train(inputs, targets, auxiliary_vars):
        return outputs_losses(
            True, inputs, targets, auxiliary_vars, self.data.losses_train
        )

    @tf.function(jit_compile=config.xla_jit)
    def outputs_losses_test(inputs, targets, auxiliary_vars):
        return outputs_losses(
            False, inputs, targets, auxiliary_vars, self.data.losses_test
        )

    opt = optimizers.get(self.opt_name, learning_rate=lr, decay=decay)

    @tf.function(jit_compile=config.xla_jit)
    def train_step(inputs, targets, auxiliary_vars):
        # inputs and targets are np.ndarray and automatically converted to Tensor.
        with tf.GradientTape() as tape:
            losses = outputs_losses_train(inputs, targets, auxiliary_vars)[1]
            total_loss = tf.math.reduce_sum(losses)
        trainable_variables = (
            self.net.trainable_variables + self.external_trainable_variables
        )

        grads = tape.gradient(total_loss, trainable_variables)
        # # Zip grads and variables
        # grads_vars = list(zip(grads, trainable_variables))

        # # Remove entries where grad is None
        # grads_vars = [(g, v) for g, v in grads_vars if g is not None]

        # print(grads)
        # Optionally warn:
        # for i, (g, v) in enumerate(zip(grads, trainable_variables)):
        #     if g is None:
        #         print(f"Warning: No gradient for variable: {v.name}")
        grads_no_nan = [
            tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
            for grad in grads
        ]
        clipped_grads = [tf.clip_by_norm(grad, 10) for grad in grads_no_nan]
        # tf.print(clipped_grads, summarize = -1)
        # opt.apply_gradients(zip(grads, trainable_variables))
        opt.apply_gradients(zip(clipped_grads, trainable_variables))
        # opt.apply_gradients(grads_vars)

    def train_step_tfp(
        inputs, targets, auxiliary_vars, previous_optimizer_results=None
    ):
        def build_loss():
            losses = outputs_losses_train(inputs, targets, auxiliary_vars)[1]
            return tf.math.reduce_sum(losses)

        trainable_variables = (
            self.net.trainable_variables + self.external_trainable_variables
        )
        return opt(trainable_variables, build_loss, previous_optimizer_results)

    # Callables
    self.outputs = outputs
    self.outputs_losses_train = outputs_losses_train
    self.outputs_losses_test = outputs_losses_test
    self.train_step = (
        train_step
        if not optimizers.is_external_optimizer(self.opt_name)
        else train_step_tfp
    )


def patch_compile_tensorflow():
    from deepxde import Model

    Model._compile_tensorflow = custom_compile_tensorflow


# Adds initializer for the bias values. Otherwise their initial
# values are zeros.
class FNN_custom(NN):
    """Fully-connected neural network."""

    def __init__(
        self,
        layer_sizes,
        activation,
        kernel_initializer,
        bias_initializer,
        regularization=None,
        dropout_rate=0,
    ):
        super().__init__()
        self.regularizer = regularizers.get(regularization)
        self.dropout_rate = dropout_rate

        self.denses = []
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            activation = list(map(activations.get, activation))
        else:
            activation = activations.get(activation)
        initializer_kernel = initializers.get(kernel_initializer)
        initializer_bias = initializers.get(bias_initializer)

        for j, units in enumerate(layer_sizes[1:-1]):
            self.denses.append(
                tf.keras.layers.Dense(
                    units,
                    activation=(
                        activation[j] if isinstance(activation, list) else activation
                    ),
                    kernel_initializer=initializer_kernel,
                    bias_initializer=initializer_bias,
                    kernel_regularizer=self.regularizer,
                )
            )
            if self.dropout_rate > 0:
                self.denses.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

        self.denses.append(
            tf.keras.layers.Dense(
                layer_sizes[-1],
                kernel_initializer=initializer_kernel,
                bias_initializer=initializer_bias,
                kernel_regularizer=self.regularizer,
            )
        )

    def call(self, inputs, training=False):
        y = inputs
        if self._input_transform is not None:
            y = self._input_transform(y)
        for f in self.denses:
            y = f(y, training=training)
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y


class FNN_custom_PositiveOutput(NN):
    """Fully-connected neural network with strictly positive outputs via softplus."""

    def __init__(
        self,
        layer_sizes,
        activation,
        kernel_initializer,
        bias_initializer,
        regularization=None,
        dropout_rate=0,
    ):
        super().__init__()
        self.regularizer = regularizers.get(regularization)
        self.dropout_rate = dropout_rate

        self.denses = []
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            activation = list(map(activations.get, activation))
        else:
            activation = activations.get(activation)

        initializer_kernel = initializers.get(kernel_initializer)
        initializer_bias = initializers.get(bias_initializer)

        # Hidden layers
        for j, units in enumerate(layer_sizes[1:-1]):
            self.denses.append(
                tf.keras.layers.Dense(
                    units,
                    activation=(
                        activation[j] if isinstance(activation, list) else activation
                    ),
                    kernel_initializer=initializer_kernel,
                    bias_initializer=initializer_bias,
                    kernel_regularizer=self.regularizer,
                )
            )
            if self.dropout_rate > 0:
                self.denses.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

        # Final layer (no activation here)
        self.output_layer = tf.keras.layers.Dense(
            layer_sizes[-1],
            kernel_initializer=initializer_kernel,
            bias_initializer=initializer_bias,
            kernel_regularizer=self.regularizer,
        )

    def call(self, inputs, training=False):
        y = inputs
        if self._input_transform is not None:
            y = self._input_transform(y)

        for f in self.denses:
            y = f(y, training=training)

        # Final output with softplus activation to enforce positivity
        y = tf.math.softplus(self.output_layer(y)) + 1e-10

        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y


def patch_fnn():
    import deepxde

    deepxde.nn.tensorflow.fnn.FNN = FNN_custom_PositiveOutput  # FNN_custom


class LogLosses(deepxde.callbacks.Callback):
    def __init__(self, save_every=1000, outdir="results", filename="losses.csv"):
        super().__init__()
        self.save_every = save_every
        self.outdir = outdir
        self.loss_file = os.path.join(outdir, filename)

    def on_train_begin(self):
        # Get the number of loss components from the first training step
        # We'll update the header after the first step to know how many losses we have
        with open(self.loss_file, "w") as f:
            f.write("step,total_loss")  # Will add individual loss columns dynamically

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.save_every == 0:
            losses = self.model.train_state.loss_train
            total = np.sum(losses)

            # If this is the first time we're logging, update the header with individual loss names
            if step == self.save_every:
                # Rewrite the header to include individual losses
                with open(self.loss_file, "w") as f:
                    header = "step,total_loss"
                    for i in range(len(losses)):
                        header += f",loss_{i+1}"
                    f.write(header + "\n")

            # Write the data
            with open(self.loss_file, "a") as f:
                line = f"{step},{total}"
                for loss in losses:
                    line += f",{loss}"
                f.write(line + "\n")


patch_compile_tensorflow()
patch_fnn()
# patch_training_display()

from deepxde.backend import set_default_backend
import os
from tensorflow import keras

set_default_backend("tensorflow")
import deepxde
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import csv
import matplotlib.pyplot as plt
import DPINNs_utils
from DPINNs_utils import LogLosses

deepxde.config.set_random_seed(42)

data, parameters = [
    ["PK_simulation_unscaled_cor_0.0.csv"],
    ["parameters_cor_0.0.csv"],
]
correlations_names = ["0"]
for index, element in enumerate(zip(parameters, data)):
    print(f"iteration number {index}")

    def PK_system(t, nn_output):
        # A refers to the drug amount
        C_mean = nn_output[:, 0:1]
        C_var = nn_output[:, 1:2]

        # Sample N values from A, ke, and Vd
        N = 100

        mu_C = tf.math.log((C_mean**2) / tf.math.sqrt((C_mean**2) + (C_var)))
        sigma_C = tf.math.sqrt(tf.math.log(1 + ((C_var) / (C_mean**2))))

        # ke and Vd are global trainable parameters (mean and sd)
        mu_ke = ke_mean
        mu_Vd = Vd_mean
        sigma_error = tf.math.softplus(sigma) + 1e-6
        sigma_ke = tf.math.softplus(ke_sd) + 1e-6
        sigma_Vd = tf.math.softplus(Vd_sd) + 1e-6

        cor_C_ke_tr = tf.math.tanh(cor_C_ke)
        cor_C_Vd_tr = tf.math.tanh(cor_C_Vd)
        cor_ke_Vd_tr = tf.math.tanh(cor_ke_Vd)
        # Correlation matrix for (A, ke, Vd)

        combo1 = tf.stack([1.0, cor_C_ke_tr, cor_C_Vd_tr])
        combo1_reshaped = tf.reshape(combo1, (3, 1))
        combo2 = tf.stack([cor_C_ke_tr, 1.0, cor_ke_Vd_tr])
        combo2_reshaped = tf.reshape(combo2, (3, 1))
        combo3 = tf.stack([cor_C_Vd_tr, cor_ke_Vd_tr, 1.0])
        combo3_reshaped = tf.reshape(combo3, (3, 1))
        corr_matrix = tf.transpose(
            tf.concat(
                [
                    combo1_reshaped,
                    combo2_reshaped,
                    combo3_reshaped,
                ],
                1,
            )
        )
        size = tf.shape(mu_C)[0]
        mu = tf.concat(
            [
                mu_C,
                tf.fill(dims=[size, 1], value=mu_ke),
                tf.fill(dims=[size, 1], value=mu_Vd),
            ],
            axis=1,
        )
        S = tf.linalg.diag(
            tf.concat(
                [
                    sigma_C,
                    tf.math.abs(tf.fill(dims=[size, 1], value=sigma_ke)),
                    tf.math.abs(tf.fill(dims=[size, 1], value=sigma_Vd)),
                ],
                axis=1,
            )
        )
        cov_mat = tf.linalg.matmul(tf.linalg.matmul(S, corr_matrix), S)

        # Add a small number to the diagonal to make the covariance matrix positive definite
        # cov_mat += 5e-4 * tf.eye(N_par)

        mvn = tfp.distributions.MultivariateNormalTriL(
            loc=mu,
            scale_tril=tfp.experimental.linalg.simple_robustified_cholesky(cov_mat),
        ).sample(N)

        error = tfp.distributions.Normal(0.0, sigma_error).sample([size, N])
        C_samples = tf.transpose(tf.exp(mvn[:, :, 0])) / tf.math.exp(error)

        ke_samples = tf.transpose(tf.exp(mvn[:, :, 1]))
        Vd_samples = tf.transpose(tf.exp(mvn[:, :, 2]))
        dC_samples = -ke_samples * C_samples
        C_bar = tf.math.reduce_mean(C_samples, axis=1, keepdims=True)

        dC_mean_denoise = tf.math.reduce_mean(dC_samples, axis=1, keepdims=True)

        dC_var_denoise = (2 / (N - 1)) * tf.math.reduce_sum(
            (C_samples - C_bar) * (dC_samples - dC_mean_denoise),
            axis=1,
            keepdims=True,
        )

        dC_mean = dC_mean_denoise * tf.math.exp((sigma_error**2) / 2)
        exp_sigma = tf.math.exp(sigma_error**2)
        dC_var = exp_sigma * (
            exp_sigma * dC_var_denoise + (exp_sigma - 1) * 2 * C_bar * dC_mean_denoise
        )

        dC_mean_AD = deepxde.grad.jacobian(nn_output, t, i=0)
        dC_var_AD = deepxde.grad.jacobian(nn_output, t, i=1)
        ode_loss1 = dC_mean_AD - dC_mean
        ode_loss2 = dC_var_AD - dC_var

        # Initial condition losses
        IC_index_tensor = tf.argmax(tf.cast(tf.less(tf.abs(t), 1e-20), tf.int32))
        IC_index = tf.squeeze(IC_index_tensor, axis=0)

        Vd_IC = Vd_samples[IC_index : IC_index + 1, :]  # shape [1, N]
        error_IC = error[IC_index : IC_index + 1, :]  # shape [1, N]

        C0_samples = (dose / Vd_IC) * tf.math.exp(error_IC)
        mean_C0 = tf.reduce_mean(C0_samples)
        var_C0 = tf.math.reduce_variance(C0_samples)

        C_mean_pred_0 = nn_output[IC_index, 0]
        C_var_pred_0 = nn_output[IC_index, 1]

        IC_loss1 = tf.where(
            tf.less(tf.abs(t), 1e-20),
            (C_mean_pred_0 - mean_C0) * tf.sqrt(tf.cast(size, tf.float32)),
            tf.zeros_like(C_mean_pred_0),
        )

        IC_loss2 = tf.where(
            tf.less(tf.abs(t), 1e-20),
            (C_var_pred_0 - var_C0) * tf.sqrt(tf.cast(size, tf.float32)),
            tf.zeros_like(C_mean_pred_0),
        )

        return [ode_loss1, ode_loss2, IC_loss1, IC_loss2]

    # Import the parameters relevant to the scaling of the data
    with open(f"{element[0]}", "r") as csvfile:
        my_reader = csv.reader(csvfile)
        header = next(my_reader)  # Skip the header row
        data_row = next(my_reader)  # Read the data row

        # Assign values to variables
        (
            t_bar,
            t_sd,
            C_mean_bar,
            C_mean_sd,
            C_var_bar,
            C_var_sd,
            t_max,
            t_min,
            C_mean_max,
            C_mean_min,
            C_var_max,
            C_var_min,
            C_max,
            C_min,
        ) = map(float, data_row)

    def boundary_initial(x, on_initial):
        return on_initial and np.isclose(x[0], 0.0)

    # Load train data
    def gen_traindata(file_name):
        data = pd.read_csv(file_name)
        return (
            np.vstack(data["t"]),
            np.vstack(data["C_mean"]),
            np.vstack(data["C_var"]),
        )

    # Organize and assign the train data
    t_obs, C_mean_obs, C_var_obs = gen_traindata(f"{element[1]}")

    observe_C_mean = deepxde.icbc.PointSetBC(t_obs, C_mean_obs, component=0)
    observe_C_var = deepxde.icbc.PointSetBC(t_obs, C_var_obs, component=1)

    # Create the data object first so we can access its collocation points
    geom = deepxde.geometry.TimeDomain(0.0, 24.0)

    dose = tf.constant(300.0, dtype=tf.float32)

    # 3. In the DeepXDE data object, set auxiliary_var_function to generate_samples
    t_dense = np.linspace(0, 24, 100).reshape(-1, 1)

    data = deepxde.data.PDE(
        geometry=geom,
        pde=PK_system,
        bcs=[
            observe_C_mean,
            observe_C_var,
        ],
        anchors=t_dense,
    )

    ke_mean = deepxde.Variable(1e-4)
    ke_sd = deepxde.Variable(1e-4)
    Vd_mean = deepxde.Variable(1e-4)
    Vd_sd = deepxde.Variable(1e-4)
    cor_C_ke = deepxde.Variable(1e-4)
    cor_C_Vd = deepxde.Variable(1e-4)
    cor_ke_Vd = deepxde.Variable(1e-4)
    sigma = deepxde.Variable(1e-4)

    external_trainable_variables = [
        ke_mean,
        ke_sd,
        Vd_mean,
        Vd_sd,
        cor_C_ke,
        cor_C_Vd,
        cor_ke_Vd,
        sigma,
    ]
    # Callback to report the kinetic parameters
    variable = deepxde.callbacks.VariableValue(
        external_trainable_variables,
        period=1000,
        filename=f"variables_correlation_{correlations_names[index]}.dat",
        precision=5,
    )

    # initializer = tf.keras.initializers.he_normal(seed=100) #"Glorot uniform"
    kernel_initializer = keras.initializers.GlorotNormal(seed=9827)
    bias_initializer = keras.initializers.GlorotNormal(seed=11)
    # initializer = tf.keras.initializers.RandomNormal(mean=0.01, stdeepxdev=0.005, seed=100)
    activation = "tanh"  # tf.keras.layers.LeakyReLU(alpha=0.5) #"relu"
    net = deepxde.nn.tensorflow.fnn.FNN(
        [1] + [3] * 2 + [2],
        activation,
        kernel_initializer,
        bias_initializer,
    )

    # Set loss_weights to match the number of loss terms: 2 ICs, 2 data, 2 PDEs
    loss_weights = [1, 1, 1, 1, 1, 1]

    # Compile the model with the optimizer following an exponential decay approach in the learning rate
    lr_init = 0.0005
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_init)

    model = deepxde.Model(data, net)
    model.compile(
        optimizer=optimizer,
        external_trainable_variables=external_trainable_variables,
        loss_weights=loss_weights,
    )

    # Ensure results directory exists before saving files
    os.makedirs("results", exist_ok=True)

    log_losses = LogLosses(
        save_every=1000,
        filename=f"losses_correlation_{correlations_names[index]}.csv",
    )

    losshistory, train_state = model.train(
        iterations=100000, callbacks=[variable, log_losses]
    )

    t_points = np.reshape(np.linspace(0, 24, 1000), (1000, 1))
    # Plot predictions of the NN
    nn_preds = model.predict(t_points)
    preds_df = pd.DataFrame(
        np.column_stack((t_points, nn_preds)),
        columns=["t", "C_mean", "C_var"],
    )
    preds_df.to_csv("predictions_cor_0.0.csv", index=False)

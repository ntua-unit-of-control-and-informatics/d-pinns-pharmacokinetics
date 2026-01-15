from deepxde.backend import set_default_backend
import os
from tensorflow import keras

set_default_backend("tensorflow")
import deepxde
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import DPINNs_utils
from DPINNs_utils import LogLosses

import matplotlib.pyplot as plt

GlobalSeed = 1111
deepxde.config.set_random_seed(GlobalSeed)
# -------------------------------------------------------------------------
# Data loading (same style as old code)
# -------------------------------------------------------------------------
mPBPK_data = pd.read_csv("mPBPK/Data/preprocessed_mPBPK_data.csv")
mPBPK_data = mPBPK_data[:-1]  # drop last row as you did
# mPBPK_data  = mPBPK_data[6:]
maximum1 = 25
minimum = 0.1
delta = maximum1 - minimum
maximum2 = 3.5
delta2 = maximum2 - minimum
t_obs = mPBPK_data["TIME"].values.reshape(-1, 1)
C_mean_obs = mPBPK_data["MEAN"].values.reshape(-1, 1)
C_mean_obs = (C_mean_obs - minimum) / delta
C_sd_obs = (mPBPK_data["SD"].values).reshape(-1, 1)
C_sd_obs = (C_sd_obs - minimum) / delta2

t_max = float(t_obs.max())

N = 100  # 300  # number of MC samples (same as old code)
Ntime = 50  # 100

# -------------------------------------------------------------------------
# Physiological constants (70 kg human)
# -------------------------------------------------------------------------
MPBPK_CONSTANTS = {
    "L": 3.0,  # Total lymph flow (L/day)
    "sigma_L": 0.2,  # Lymphatic capillary reflection coefficient
    "Kp": 0.8,  # Available ISF fraction (fixed)
}


def mPBPK_system(t, nn_output):
    """
    D-PINN mPBPK system with the same logic as the old 1-compartment model:
    - NN outputs plasma mean & variance (plus 3 other means)
    - C_mean, C_var -> lognormal (mu_C, sigma_C)
    - joint MVN sampling of [log C_plasma, log CLp]
    - ODEs for Cp, Ctight, Cleaky, Clymph
    - MC estimates of dCp_mean, dCp_var compared to AD derivatives
    """
    # Small epsilon for stability
    eps = 1e-10

    # ---------------------------------------------------------------------
    # 1. Extract NN outputs
    #    nn_output: [batch_size, 5] = [Cp_mean, Cp_var, Ct_mean, Cl_mean, Clymph_mean]
    # ---------------------------------------------------------------------
    C_mean = nn_output[:, 0:1] * delta + minimum  # plasma mean [size,1]
    C_var = (nn_output[:, 1:2] * delta2 + minimum) ** 2
    C_sd = tf.math.sqrt(C_var)  # plasma sd [size,1]

    # tight, leaky, lymph means (deterministic, no variance modelled)
    Ctight_mean = nn_output[:, 2:3] * delta + minimum  # [size,1]
    Cleaky_mean = nn_output[:, 3:4] * delta + minimum  # [size,1]
    Clymph_mean = nn_output[:, 4:5] * delta + minimum  # [size,1]

    # ---------------------------------------------------------------------
    # 2. Convert plasma mean/var to lognormal (mu_C, sigma_C)
    #    (same formulas as old code)
    # ---------------------------------------------------------------------
    C_mean_safe = C_mean + eps
    C_var_safe = C_var + eps

    mu_C = tf.math.log((C_mean_safe**2) / tf.math.sqrt(C_mean_safe**2 + C_var_safe))
    sigma_C = tf.math.sqrt(tf.math.log(1.0 + C_var_safe / (C_mean_safe**2)))

    # ---------------------------------------------------------------------
    # 3. Trainable parameters
    # ---------------------------------------------------------------------
    sigma1 = tf.nn.sigmoid(sigma1_raw)  # Maps to (0,1)
    sigma2 = sigma1 * tf.nn.sigmoid(sigma2_raw)  # Ensures σ2 < σ1

    # CLp lognormal parameters
    mu_CLp = CLp_mean  # scalar
    sigma_CLp = tf.math.softplus(CLp_sd) + 1e-10

    # Observation error (FIXED at 10%)
    sigma_error = sigma_error_fixed

    # Correlation between log C and log CLp
    rho_raw = cor_C_CLp
    rho = tf.math.tanh(rho_raw)  # in (-1,1)
    # rho = tf.constant(-1.0) #
    # ---------------------------------------------------------------------
    # 4. Build joint MVN for [log C_plasma, log CLp] (batched over time points)
    # ---------------------------------------------------------------------
    size = tf.shape(mu_C)[0]

    # Means: [mu_C(t_i), mu_CLp] for each time point
    mu = tf.concat(
        [
            mu_C,  # [size,1]
            tf.fill(dims=[size, 1], value=mu_CLp),  # [size,1]
        ],
        axis=1,
    )  # [size, 2]

    # Standard deviations on diagonal
    S = tf.linalg.diag(
        tf.concat(
            [
                sigma_C,  # [size,1]
                tf.fill(dims=[size, 1], value=sigma_CLp),  # [size,1]
            ],
            axis=1,
        )
    )  # [size,2,2]

    # 2x2 correlation matrix with rho
    corr_matrix = tf.reshape(tf.stack([1.0, rho, rho, 1.0]), (2, 2))  # [2,2]

    # Batched covariance
    cov_mat = tf.linalg.matmul(tf.linalg.matmul(S, corr_matrix), S)  # [size,2,2]
    N_par = tf.shape(cov_mat)[-1]
    cov_mat += 5e-7 * tf.eye(N_par, dtype=tf.float32)

    # MVN sampling: [N, size, 2] -> transpose to [size, N, 2]
    mvn = tfp.distributions.MultivariateNormalTriL(
        loc=mu,
        scale_tril=tfp.experimental.linalg.simple_robustified_cholesky(cov_mat),
    ).sample(N)  # [N, size, 2]

    mvn_t = tf.transpose(mvn, [1, 0, 2])  # [size, N, 2]

    # ---------------------------------------------------------------------
    # 5. Transform to C and CLp samples, add error (multiplicative log-normal error)
    # ---------------------------------------------------------------------
    # Sample error from Normal(0, sigma_error)
    error = tfp.distributions.Normal(0.0, sigma_error).sample([size, N])

    # Apply exp transform and error term to concentration samples
    C_samples_mvn = tf.exp(mvn_t[:, :, 0])  # [size, N]
    C_samples = C_samples_mvn / tf.math.exp(error)  # [size, N] - divide by exp(error)
    CLp_samples = tf.exp(mvn_t[:, :, 1])  # [size, N]

    # Build full compartment concentration samples
    Cp_samples = C_samples  # [size,N]
    Ctight_samples = tf.tile(Ctight_mean, [1, N])  # [size,N]
    Cleaky_samples = tf.tile(Cleaky_mean, [1, N])  # [size,N]
    Clymph_samples = tf.tile(Clymph_mean, [1, N])  # [size,N]

    # Stack into [size, N, 4]
    C_samples_full = tf.stack(
        [Cp_samples, Ctight_samples, Cleaky_samples, Clymph_samples],
        axis=-1,
    )  # [size,N,4]

    # ---------------------------------------------------------------------
    # 6. mPBPK ODEs for each MC sample (same equations as your new code)
    # ---------------------------------------------------------------------
    # Sample bodyweight from uniform distribution (50-75 kg)

    BW_samples = tf.random.uniform(
        shape=[tf.shape(t)[0], N], minval=50.0, maxval=75.0
    )  # [size,N]
    # Estimate blood volume as fraction of bodyweight
    Vblood_fraction = 0.0733  # fraction of bodyweight (L/kg) FELDSCHUH, et al. 1977
    Vblood_samples = BW_samples * Vblood_fraction  # L (blood volume)

    # Estimate Lymph volume assuming to be equal to blood volume
    Vlymph_samples = Vblood_samples  # L

    # Estimate plasma volume based on hematocrit
    hematocrit = 0.43  # average human hematocrit Hsu et al. 2001
    Vp_samples = Vblood_samples * (1 - hematocrit)  # L

    # Calculate ISF volumes as fraction of bodyweight
    ISF_fraction = 0.15  # fixed fraction of bodyweight for ISF
    ISF_total = BW_samples * ISF_fraction * MPBPK_CONSTANTS["Kp"]  # L (total ISF)
    Vtight_samples = 0.65 * ISF_total  # L (tight tissues ISF)
    Vleaky_samples = 0.35 * ISF_total  # L (leaky tissues ISF)

    # Sample lymph flow fraction from normal distribution
    L_total = tf.fill([size, N], 3.0 / 24.0)  # L/h Stucker et al. 2008

    # Calculate dose as 1 mg/kg bodyweight
    dose_mg = BW_samples * 1.0  # mg (1 mg/kg)

    # Fixed parameters (convert L/day to L/h)
    # Calculate L1 and L2 from sampled lymph flow (maintaining 33%/67% split)
    L1_h = 0.33 * L_total  # L/day (tight tissue lymph flow)
    L2_h = 0.67 * L_total  # L/day (leaky tissue lymph flow)

    sigma_L = tf.fill([size, N], MPBPK_CONSTANTS["sigma_L"])

    # Input function: 90-minute IV infusion (1 mg/kg)
    infusion_duration = 1.5  # 90 minutes = 1.5 hours

    # Infusion rate: dose_mg / infusion_duration during infusion period
    infusion_rate = dose_mg / infusion_duration

    # Apply infusion rate during [0, 1.5] hours, zero afterwards
    input_rate = tf.where(
        tf.logical_and(t >= 0.0, t <= infusion_duration), infusion_rate, 0.0
    )

    # mPBPK Model A differential equations
    dCp_dt_phys = (
        input_rate
        + Clymph_samples * L_total
        - Cp_samples * L1_h * (1 - sigma1)
        - Cp_samples * L2_h * (1 - sigma2)
        - CLp_samples * Cp_samples
    ) / Vp_samples

    dCtight_dt_phys = (
        L1_h * (1 - sigma1) * Cp_samples - L1_h * (1 - sigma_L) * Ctight_samples
    ) / Vtight_samples

    dCleaky_dt_phys = (
        L2_h * (1 - sigma2) * Cp_samples - L2_h * (1 - sigma_L) * Cleaky_samples
    ) / Vleaky_samples

    dClymph_dt_phys = (
        L1_h * (1 - sigma_L) * Ctight_samples
        + L2_h * (1 - sigma_L) * Cleaky_samples
        - Clymph_samples * L_total
    ) / Vlymph_samples

    # Stack derivatives [size,N,4]
    dC_samples_full = tf.stack(
        [dCp_dt_phys, dCtight_dt_phys, dCleaky_dt_phys, dClymph_dt_phys],
        axis=-1,
    )
    # ---------------------------------------------------------------------
    # 7. MC-based mean/variance derivatives with error propagation
    # ---------------------------------------------------------------------
    # Mean over samples (denoised)
    C_bar_full = tf.reduce_mean(C_samples_full, axis=1)  # [size, 4]
    dC_mean_denoise = tf.reduce_mean(dC_samples_full, axis=1)  # [size, 4]

    # ----- Plasma variance derivative (denoised) -----
    Cpl_samples = C_samples_full[:, :, 0]  # [size, N]
    dCpl_samples = dC_samples_full[:, :, 0]  # [size, N]
    Cpl_bar = C_bar_full[:, 0:1]  # [size, 1]
    Cpl_bar_N = tf.tile(Cpl_bar, [1, N])  # shape [size, N]
    dC_mean_pl_denoise = dC_mean_denoise[:, 0:1]  # [size, 1]
    dC_mean_pl_N = tf.tile(dC_mean_pl_denoise, [1, N])  # shape [size, N]
    C_mean_N = tf.tile(C_mean, [1, N])  # shape [size, N]
    C_var_N = tf.tile(C_var, [1, N])
    C_sd_N = tf.tile(C_sd, [1, N])

    dC_var_denoise = (2.0 / (N - 1)) * tf.reduce_sum(
        (Cpl_samples - Cpl_bar_N) * (dCpl_samples - dC_mean_pl_N),
        axis=1,
        keepdims=True,
    )  # [size, 1]

    # ---------------------------------------------------------------------
    # ERROR PROPAGATION: Apply multiplicative log-normal error corrections
    # ---------------------------------------------------------------------
    # Mean derivative correction: multiply by exp(sigma_error^2 / 2)
    dC_mean_pl_corrected = dC_mean_pl_denoise * tf.math.exp((sigma_error**2) / 2)

    # Variance derivative correction
    exp_sigma = tf.math.exp(sigma_error**2)
    dC_var_corrected = exp_sigma * (
        exp_sigma * dC_var_denoise + (exp_sigma - 1) * 2 * Cpl_bar * dC_mean_pl_denoise
    )  # [size, 1]

    # Convert variance derivative to SD derivative: d(SD)/dt = d(Var)/dt / (2*SD)
    dC_sd_corrected = dC_var_corrected / (2 * C_sd + eps)  # [size, 1]

    # Other compartments: use denoised derivatives directly (no error propagation)
    dCtight_mean_true = dC_mean_denoise[:, 1:2]  # [size, 1]
    dCleaky_mean_true = dC_mean_denoise[:, 2:3]  # [size, 1]
    dClymph_mean_true = dC_mean_denoise[:, 3:4]  # [size, 1]

    # ---------------------------------------------------------------------
    # 8. Automatic differentiation of NN outputs & ODE residuals
    # ---------------------------------------------------------------------
    dCp_mean_AD = deepxde.grad.jacobian(nn_output, t, i=0) * delta  # [size, 1]
    dCp_sd_AD = deepxde.grad.jacobian(nn_output, t, i=1) * delta2**2  # [size, 1]
    dCtight_mean_AD = deepxde.grad.jacobian(nn_output, t, i=2) * delta  # [size, 1]
    dCleaky_mean_AD = deepxde.grad.jacobian(nn_output, t, i=3) * delta  # [size, 1]
    dClymph_mean_AD = deepxde.grad.jacobian(nn_output, t, i=4) * delta  # [size, 1]

    # ODE losses: compare AD derivatives with error-corrected physics derivatives
    ode_losses = [
        tf.reshape(
            dCp_mean_AD - dC_mean_pl_corrected, [-1]
        ),  # Plasma mean with error correction
        tf.reshape(
            dCp_sd_AD - dC_sd_corrected, [-1]
        ),  # Plasma SD with error correction
        tf.reshape(
            dCtight_mean_AD - dCtight_mean_true, [-1]
        ),  # Other compartments (no error)
        tf.reshape(dCleaky_mean_AD - dCleaky_mean_true, [-1]),
        tf.reshape(dClymph_mean_AD - dClymph_mean_true, [-1]),
    ]

    return ode_losses


def boundary_initial(x, on_initial):
    return on_initial and np.isclose(x[0], 0.0)


geom = deepxde.geometry.TimeDomain(0.0, t_max)

observe_C_mean = deepxde.icbc.PointSetBC(t_obs, C_mean_obs, component=0)
observe_C_sd = deepxde.icbc.PointSetBC(t_obs, C_sd_obs, component=1)

ic_bcs = []

t_ic = np.array([[0.0]])  # time = 0
C0 = np.array([[0.0]])  # concentration = 0

# components: plasma mean (0), plasma variance (0), tight mean (2), leaky mean (3), lymph mean (4)
for comp in [0, 1, 2, 3, 4]:
    ic_bc = deepxde.icbc.PointSetBC(t_ic, C0, component=comp)
    ic_bcs.append(ic_bc)

# Collocation points (dense early, sparse late)
n1 = int(0.0 * Ntime)
n2 = int(0.5 * Ntime)
n3 = Ntime - n1 - n2
t1 = np.linspace(0.0, 200.0, n1, endpoint=True)  # very dense around distribution phase
t2 = np.linspace(200.0, 500.0, n2, endpoint=False)  # moderate density
t3 = np.linspace(500.0, t_max, n3, endpoint=True)  # sparse in the tail

t_colloc = np.concatenate([t1, t2, t3])  # shape (200, 1)
t_dense = t_colloc.reshape(-1, 1)
t_all = np.unique(np.concatenate([t_dense.flatten(), t_obs.flatten()]))
t_all = t_all.reshape(-1, 1)


N_anchor = Ntime
exp_factor = 1.0
u = np.linspace(0.0, 1.0, N_anchor)
t_dense = (u**exp_factor) * t_max
t_dense = t_dense.reshape(-1, 1)
t_extra = t_dense  # from whatever scheme
t_all = np.unique(np.concatenate([t_extra.flatten(), t_obs.flatten()]))
t_all = t_all.reshape(-1, 1)

data = deepxde.data.PDE(
    geometry=geom,
    pde=mPBPK_system,
    bcs=[observe_C_mean, observe_C_sd] + ic_bcs,
    anchors=t_all,
)

# -------------------------------------------------------------------------
# Trainable variables (template-style)
# -------------------------------------------------------------------------
print("Initializing trainable parameters...")

# σ1, σ2 parameters with hierarchical constraint
sigma1_raw = deepxde.Variable(0.0)  # Will map to σ1 ∈ (0,1)
sigma2_raw = deepxde.Variable(0.0)  # Will ensure σ2 < σ1

# CLp parameter (lognormal)
CLp_mean = deepxde.Variable(0.0)
CLp_sd = deepxde.Variable(0.0)

# Correlation parameter
cor_C_CLp = deepxde.Variable(0.0)

# Error term - FIXED at 10%
sigma_error_fixed = 0.10

external_trainable_variables = [
    sigma1_raw,
    sigma2_raw,
    CLp_mean,
    CLp_sd,
    cor_C_CLp,
]

print("Trainable parameters initialized:")
print("- σ1 (tight tissues reflection coefficient)")
print("- σ2 (leaky tissues reflection coefficient)")
print("- CLp (plasma clearance rate)")
print("- ρ(C, CLp) (correlation between concentration and clearance)")
print("\nFixed parameters:")
print(f"- Error term (sigma_error): {sigma_error_fixed} (10%)")

variable_cb = deepxde.callbacks.VariableValue(
    external_trainable_variables,
    period=1000,
    filename="mPBPK/results/mPBPK_dpinn_parameters.dat",
    precision=5,
)

# -------------------------------------------------------------------------
# Network & training (same style as old code)
# -------------------------------------------------------------------------
# kernel_initializer = keras.initializers.GlorotNormal(seed=GlobalSeed)
# bias_initializer   = keras.initializers.GlorotNormal(seed=GlobalSeed)
kernel_initializer = keras.initializers.LecunNormal(seed=GlobalSeed)
bias_initializer = keras.initializers.Zeros()
activation = "tanh"

net = deepxde.nn.tensorflow.fnn.FNN(
    [1] + 2 * [30] + [5],
    activation,
    kernel_initializer,
    bias_initializer,
)

ode_weights = [1] * 5
data_weights = [1, 1]
ic_weights = [1] * 5
loss_weights = ode_weights + data_weights + ic_weights

model = deepxde.Model(data, net)

initial_lr = 5e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=1000,
    decay_rate=0.99,
    staircase=True,
)


model.compile(
    "adam",
    lr=lr_schedule,
    loss="mse",
    external_trainable_variables=external_trainable_variables,
    loss_weights=loss_weights,
)


class LossSummaryCallback(deepxde.callbacks.Callback):
    def __init__(self, period=1000):
        super().__init__()
        self.period = period

    def on_epoch_end(self):
        if self.model.train_state.epoch % self.period == 0:
            losses = self.model.train_state.loss_train
            ode_total = np.sum(losses[:5])
            data_total = np.sum(losses[5:7])
            ic_total = np.sum(losses[7:12])
            total_loss = np.sum(losses)
            print(
                f"Epoch {self.model.train_state.epoch}: "
                f"Total={total_loss:.2e}, ODE={ode_total:.2e}, "
                f"Data={data_total:.2e}, IC={ic_total:.2e}"
            )


os.makedirs(
    "mPBPK/results",
    exist_ok=True,
)

log_losses = LogLosses(
    save_every=1000,
    outdir="mPBPK/results",
    filename="mPBPK_dpinn_losses.csv",
)

print("Starting training...")
iterations = 70000
loss_summary = LossSummaryCallback(period=1000)

losshistory, _ = model.train(
    iterations=iterations,
    callbacks=[variable_cb, loss_summary, log_losses],
)


# -------------------------------------------------------------------------
# Save predictions on observation grid
# -------------------------------------------------------------------------
y_pred = model.predict(t_obs / t_max)
pd.DataFrame(
    {
        "TIME": t_obs.flatten(),
        "MEAN_OBSERVED": C_mean_obs.flatten(),
        "MEAN_PREDICTED": y_pred[:, 0],
        "SD_OBSERVED": C_sd_obs.flatten(),
        "SD_PREDICTED": y_pred[:, 1],
    }
).to_csv(
    "mPBPK/results/mPBPK_dpinn_predictions_results.csv",
    index=False,
)


# -------------------------------------------------------------------------
# Save predictions on observation grid+
# -------------------------------------------------------------------------
# Predict at observation times
y_pred_obs = model.predict(t_obs)  # shape (N_obs, 5)

t = t_obs.flatten()

# Dense time grid for smooth curves
t_points = np.linspace(0, t_max, 10000).reshape(-1, 1)
nn_out = model.predict(t_points)  # shape should be (10000, 5)
n_outputs = nn_out.shape[1]

# Observed data (plasma)
mean_obs = C_mean_obs.flatten()
sd_obs = C_sd_obs.flatten()  # or mPBPK_data["SD"].values

# Predicted SD at observation times (from variance = output 1)
sd_pred = y_pred_obs[:, 1]

fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# Labels for the 5 outputs (we'll skip index 1 in the first panel)
output_labels = [
    "Plasma concentration",  # 0
    "Plasma variance",  # 1 (not used in panel 1)
    "Tight tissues",  # 2
    "Leaky tissues",  # 3
    "Lymph",  # 4
]

# ---------- Panel 1: concentrations in all compartments ----------
ax = axes[0]
for i in [0, 2, 3, 4]:
    ax.plot(
        t_points.flatten(),
        nn_out[:, i],
        label=output_labels[i],
        linewidth=2,
    )

# Overlay observed plasma mean
ax.scatter(t, mean_obs, label="Observed plasma mean", color="black", s=25)

ax.set_ylabel("Concentration")
ax.set_title("Predicted concentrations in all compartments")
ax.legend()
ax.grid(True)

# ---------- Panel 2: predicted vs observed SD (plasma) ----------
ax = axes[1]
ax.plot(t, sd_pred, label="Predicted SD", linewidth=2)
ax.scatter(t, sd_obs, label="Observed SD", color="black", s=25)

ax.set_xlabel("Time")
ax.set_ylabel("Standard deviation")
ax.set_title("Predicted vs observed SD (plasma)")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig(
    "mPBPK/results/mPBPK_dpinn_timecourse_mean_sd_results.png",
    dpi=300,
)
# plt.show()

# -------------------------------------------------------------------------
# Print Final Parameter Estimates in Normal Scale
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL PARAMETER ESTIMATES (Normal Scale)")
print("=" * 70)

# Extract and transform parameters
final_sigma1 = tf.nn.sigmoid(sigma1_raw).numpy()
final_sigma2 = final_sigma1 * tf.nn.sigmoid(sigma2_raw).numpy()
final_CLp_median = np.exp(CLp_mean.numpy())

# Calculate CLp statistics in normal scale from lognormal parameters
mu_CLp_final = CLp_mean.numpy()
sigma_CLp_final = tf.math.softplus(CLp_sd).numpy() + 1e-10

# Lognormal distribution statistics
CLp_mean_normal = np.exp(mu_CLp_final + sigma_CLp_final**2 / 2)
CLp_sd_normal = np.sqrt(
    (np.exp(sigma_CLp_final**2) - 1) * np.exp(2 * mu_CLp_final + sigma_CLp_final**2)
)

# Correlation parameter
final_cor_C_CLp_raw = cor_C_CLp.numpy()
final_cor_C_CLp = np.tanh(final_cor_C_CLp_raw)

print("\nReflection Coefficients:")
print(f"  σ1 (tight tissues):  {final_sigma1:.6f}")
print(f"  σ2 (leaky tissues):  {final_sigma2:.6f}")
print(f"  Constraint σ1 > σ2:  {'✓ PASS' if final_sigma1 > final_sigma2 else '✗ FAIL'}")

print("\nPlasma Clearance (CLp):")
print(f"  Median:              {final_CLp_median:.6f} L/h")
print(f"  Mean:                {CLp_mean_normal:.6f} L/h")
print(f"  Std Dev:             {CLp_sd_normal:.6f} L/h")
print(f"  CV:                  {(CLp_sd_normal/CLp_mean_normal)*100:.2f}%")

print("\nCorrelation:")
print(f"  ρ(C, CLp):           {final_cor_C_CLp:.6f}")

print("\nObservation Error (FIXED):")
print(f"  σ_error:             {sigma_error_fixed:.6f} (10%)")

print("\nPhysiological Interpretation:")
print(
    f"  (1 - σ1):            {1-final_sigma1:.6f} → {(1-final_sigma1)*100:.2f}% plasma→tight"
)
print(
    f"  (1 - σ2):            {1-final_sigma2:.6f} → {(1-final_sigma2)*100:.2f}% plasma→leaky"
)

# Estimate half-life (assuming typical Vp ~ 3L)
Vp_typical = 3.0  # L
half_life_hours = 0.693 * Vp_typical / final_CLp_median
print(f"\nEstimated Plasma Half-life (assuming Vp={Vp_typical}L):")
print(
    f"  t½:                  {half_life_hours:.2f} hours ({half_life_hours/24:.2f} days)"
)

print("\n" + "=" * 70)

# -------------------------------------------------------------------------
# Save Configuration Report
# -------------------------------------------------------------------------
print("\nSaving configuration report...")

# Extract network architecture dynamically
# layer_sizes = [net.layer_sizes[i] for i in range(len(net.layer_sizes))]

# Get output directory from log_losses (same directory as other results)
output_dir = log_losses.outdir
report_file = os.path.join(output_dir, "run_configuration.txt")

with open(report_file, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("D-PINN mPBPK Run Configuration Report\n")
    f.write("=" * 70 + "\n\n")

    f.write("SAMPLING PARAMETERS\n")
    f.write("-" * 70 + "\n")
    f.write(f"N (MC samples per evaluation):      {N}\n")
    f.write(f"Ntime (collocation points):          {Ntime}\n")
    f.write(f"exp_factor (time grid density):      {exp_factor}\n\n")

    f.write("FIXED PARAMETERS\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Observation error (sigma_error):     {sigma_error_fixed:.6f} (10% CV)\n\n"
    )

    f.write("LOSS WEIGHTS\n")
    f.write("-" * 70 + "\n")
    f.write(f"ODE loss weights:                    {ode_weights}\n")
    f.write(f"Data loss weights:                   {data_weights}\n")
    f.write(f"IC loss weights:                     {ic_weights}\n")
    f.write(f"Total loss terms:                    {len(loss_weights)}\n\n")

    f.write("NEURAL NETWORK ARCHITECTURE\n")
    f.write("-" * 70 + "\n")
    # f.write(f"Architecture (layer sizes):          {layer_sizes}\n")
    f.write(f"Activation function:                 {activation}\n")
    # f.write(f"Number of outputs:                   {layer_sizes[-1]}\n\n")

    f.write("OPTIMIZER CONFIGURATION\n")
    f.write("-" * 70 + "\n")
    f.write(f"Optimizer:                           Adam\n")
    f.write(f"Initial learning rate:               {initial_lr}\n")
    f.write(f"Decay steps:                         {lr_schedule.decay_steps}\n")
    f.write(f"Decay rate:                          {lr_schedule.decay_rate}\n")
    f.write(f"Total iterations:                    {iterations}\n\n")

    f.write("ESTIMATED PARAMETERS (Normal Scale)\n")
    f.write("-" * 70 + "\n")
    f.write(f"σ1 (tight tissues):                  {final_sigma1:.6f}\n")
    f.write(f"σ2 (leaky tissues):                  {final_sigma2:.6f}\n")
    f.write(
        f"Constraint σ1 > σ2:                  {'PASS ✓' if final_sigma1 > final_sigma2 else 'FAIL ✗'}\n\n"
    )

    f.write(f"CLp median [L/h]:                    {final_CLp_median:.6f}\n")
    f.write(f"CLp mean [L/h]:                      {CLp_mean_normal:.6f}\n")
    f.write(f"CLp std dev [L/h]:                   {CLp_sd_normal:.6f}\n")
    f.write(
        f"CLp CV [%]:                          {(CLp_sd_normal/CLp_mean_normal)*100:.2f}\n\n"
    )

    f.write(f"ρ(C, CLp) correlation:               {final_cor_C_CLp:.6f}\n\n")

    f.write("PHYSIOLOGICAL INTERPRETATION\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"(1 - σ1):                            {1-final_sigma1:.6f} ({(1-final_sigma1)*100:.2f}% plasma→tight)\n"
    )
    f.write(
        f"(1 - σ2):                            {1-final_sigma2:.6f} ({(1-final_sigma2)*100:.2f}% plasma→leaky)\n"
    )
    f.write(
        f"Estimated t½ (Vp={Vp_typical}L):            {half_life_hours:.2f} hours ({half_life_hours/24:.2f} days)\n\n"
    )

    f.write("=" * 70 + "\n")

print(f"Configuration report saved to: {report_file}")

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import csv
import time


# Define 1-compartment ODEs
def PK(t, y, ke):
    dydt = [-ke * y[0]]
    return dydt


for cor in [0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]:
    Vd_mean = 15  # L
    ke_mean = np.log(2) / 6  # assume half-life = 6 hours
    # Estimate the standard deviation  of these parameters based on CV = sd/mean
    CV_Vd = 0.1
    CV_ke = 0.4
    Vd_sd = CV_Vd * Vd_mean
    ke_sd = CV_ke * ke_mean

    mu_Vd = np.log((Vd_mean**2) / np.sqrt((Vd_mean**2) + (Vd_sd**2)))
    sigma_Vd = np.sqrt(np.log(1 + ((Vd_sd**2) / (Vd_mean**2))))

    mu_ke = np.log((ke_mean**2) / np.sqrt((ke_mean**2) + (ke_sd**2)))
    sigma_ke = np.sqrt(np.log(1 + ((ke_sd**2) / (ke_mean**2))))

    # Sample 20 virtual patients
    N_samples = 30
    mu = np.array([mu_ke, mu_Vd])
    S = np.diag([sigma_ke, sigma_Vd])

    # Assume negative correlation between Ke and Vd
    cor_matrix = np.array([[1, cor], [cor, 1]])

    cov_matrix = S @ cor_matrix @ S

    np.random.seed(1)
    mvn_samples = np.random.multivariate_normal(mu, cov_matrix, N_samples)

    ke_samples = np.exp(mvn_samples[:, 0])
    Vd_samples = np.exp(mvn_samples[:, 1])

    params = pd.DataFrame({"ke": ke_samples, "Vd": Vd_samples})
    # Set the initial condition
    y0 = [300]  # mg

    # Monitor patients for 24 hours
    t_span = (0, 24.0)
    t_eval_generic = np.linspace(0, 24, 1000)
    sampling_times = np.array([0.5, 1, 2, 6, 12, 24])
    t_eval = np.sort(np.unique(np.concat([t_eval_generic, sampling_times])))
    n_timepoints = len(t_eval)
    np.random.seed(999)
    noise_matrix = np.random.normal(0, 0.1, size=(n_timepoints, N_samples))
    preds = []

    for i in range(N_samples):
        ke = params["ke"][i]
        Vd = params["Vd"][i]

        sol = solve_ivp(PK, t_span, y0, args=(ke,), t_eval=t_eval)
        # Assume proportional error 10%
        C_true = sol.y[0] / Vd
        error = noise_matrix[:, i].reshape(-1, 1)
        C_obs = C_true.reshape(-1, 1) * np.exp(error)
        preds.append(C_obs)

    preds_total = np.column_stack(preds)

    mask = np.isclose(t_eval[:, None], sampling_times).any(axis=1).tolist()
    preds = preds_total[mask]

    # Estimate the mean and the variance of each output variable
    C_mean = np.mean(preds, axis=1)
    C_var = np.var(preds, axis=1)

    C_upper_quantile = np.nanpercentile(preds[:, 1:], 97.5, axis=1)
    C_lower_quantile = np.nanpercentile(preds[:, 1:], 2.5, axis=1)

    # Save the noisy data in a csv file
    df = pd.DataFrame({"t": sampling_times, "C_mean": C_mean, "C_var": C_var})

    df.to_csv(f"PK_simulation_unscaled_cor_{cor}.csv", index=False)

    t_bar = np.mean(sampling_times)
    t_sd = np.std(sampling_times)
    C_mean_bar = np.mean(C_mean)
    C_mean_sd = np.std(C_mean)
    C_var_bar = np.mean(C_var)
    C_var_sd = np.std(C_var)

    t_max = np.max(sampling_times)
    t_min = np.min(sampling_times)
    C_mean_max = np.max(C_mean)
    C_mean_min = np.min(C_mean)
    C_var_max = np.max(C_var)
    C_var_min = np.min(C_var)

    # Estimate the mean and the variance of each output variable
    C_max = np.max(preds)
    C_min = np.min(preds)

    with open(f"parameters_cor_{cor}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "t_bar",
                "t_sd",
                "_mean_bar",
                "C_mean_sd",
                "C_var_bar",
                "C_var_sd",
                "t_max",
                "t_min",
                "C_mean_max",
                "C_mean_min",
                "C_var_max",
                "C_var_min",
                "C_max",
                "C_min",
            ]
        )
        writer.writerow(
            [
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
            ]
        )

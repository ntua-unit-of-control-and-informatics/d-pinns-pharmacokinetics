"""
Comparison Plot: D-PINN vs Stan (Fixed 10% Error)

This script creates a comparison plot between:
1. D-PINN predictions (fixed 10% error version) - evaluated at experimental time points only
2. Stan results (fixed 10% error) - Monte Carlo simulations with sigma1, sigma2 fixed at mean values

Both methods:
- Use fixed 10% proportional error (sigma_error = 0.1)
- Sample CLp from lognormal distribution
- Use fixed sigma1, sigma2 values
- Evaluate only at experimental time points
- Generate median and 95% prediction intervals

Run from D-PINNs/ base directory
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Ensure we're in the right directory
if not os.path.exists("mPBPK"):
    raise RuntimeError("Please run this script from the D-PINNs base directory")

# Number of samples for Monte Carlo
N_SAMPLES = 1000

# Pre-generate fixed BW samples for reproducibility (same for both methods)
np.random.seed(42)
BW_SAMPLES_FIXED = np.random.uniform(50.0, 75.0, size=N_SAMPLES)

# Fixed physiological parameters for mPBPK model
MPBPK_CONSTANTS = {
    "L": 3.0,  # Total lymph flow (L/day) - fixed
    "sigma_L": 0.2,  # Lymphatic capillary reflection coefficient
    "Kp": 0.8,  # Available ISF fraction (fixed)
    "BW": BW_SAMPLES_FIXED,  # Fixed BW samples (kg)
}

# Fixed 10% proportional error for both methods
SIGMA_ERROR_FIXED = 0.1

print("D-PINN vs Stan Comparison (Fixed 10% Error)")
print("=" * 60)
print(
    f"Pre-generated {N_SAMPLES} fixed BW samples: {BW_SAMPLES_FIXED.min():.1f}-{BW_SAMPLES_FIXED.max():.1f} kg"
)


def extract_dpinn_results(dpinn_param_file):
    """
    Extract D-PINN parameter estimates (sigma_error is FIXED at 10%)

    Returns:
        dict: Contains final parameter estimates
    """

    if not os.path.exists(dpinn_param_file):
        raise FileNotFoundError(f"Parameter file not found: {dpinn_param_file}")

    print(f"\nReading D-PINN results from: {dpinn_param_file}")
    with open(dpinn_param_file, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError("Parameter file is empty or invalid")

    # Parse the last line (final parameter values)
    last_line = lines[-1].strip()

    if "[" in last_line and "]" in last_line:
        parts = last_line.split(" ", 1)
        iteration = int(parts[0])

        param_str = parts[1].strip()
        param_str = param_str.replace("[", "").replace("]", "")
        param_values = [float(x.strip()) for x in param_str.split(",")]

        # Accept either 4 or 5 parameters (5th is ignored if present)
        if len(param_values) not in [4, 5]:
            raise ValueError(f"Expected 4 or 5 parameters, got {len(param_values)}")

        # Extract and transform parameters (only use first 4, sigma_error is fixed)
        sigma1_raw = param_values[0]
        sigma2_factor_raw = param_values[1]
        CLp_mean = param_values[2]
        CLp_sd = param_values[3]
        # param_values[4] is ignored if present (not used, sigma_error is fixed)

        sigma1 = 1.0 / (1.0 + np.exp(-sigma1_raw))
        sigma2 = sigma1 * (1.0 / (1.0 + np.exp(-sigma2_factor_raw)))
        CLp_sd_transformed = np.log(1 + np.exp(CLp_sd)) + 1e-4
        CLp_median = np.exp(CLp_mean)

        print(f"D-PINN estimates (iteration {iteration}):")
        print(f"  σ1: {sigma1:.4f} (fixed)")
        print(f"  σ2: {sigma2:.4f} (fixed)")
        print(f"  CLp median: {CLp_median:.6f} L/h")
        print(f"  σ_error: {SIGMA_ERROR_FIXED:.4f} (FIXED at 10%)")

        return {
            "sigma1": sigma1,
            "sigma2": sigma2,
            "CLp_mean": CLp_mean,
            "CLp_sd": CLp_sd_transformed,
            "sigma_error": SIGMA_ERROR_FIXED,
        }
    else:
        raise ValueError("Could not parse parameter file format")


def extract_stan_results(stan_file):
    """
    Extract Stan parameter estimates from results file

    Returns:
        dict: Contains Stan parameter estimates
    """

    if not os.path.exists(stan_file):
        raise FileNotFoundError(f"Stan results file not found: {stan_file}")

    print(f"\nReading Stan results from: {stan_file}")

    stan_data = pd.read_csv(stan_file)

    # Extract mean values for sigma1 and sigma2 (these are fixed)
    sigma1_mean = stan_data[stan_data["parameter"] == "sigma1"]["mean"].values[0]
    sigma2_mean = stan_data[stan_data["parameter"] == "sigma2"]["mean"].values[0]

    # Extract mu_CL and tau_CL for lognormal sampling
    mu_CL = stan_data[stan_data["parameter"] == "mu_CL"]["mean"].values[0]
    tau_CL = stan_data[stan_data["parameter"] == "tau_CL"]["mean"].values[0]

    print(f"Stan estimates:")
    print(f"  σ1: {sigma1_mean:.4f} (fixed at mean)")
    print(f"  σ2: {sigma2_mean:.4f} (fixed at mean)")
    print(f"  mu_CL: {mu_CL:.6f}")
    print(f"  tau_CL: {tau_CL:.6f}")
    print(f"  σ_error: {SIGMA_ERROR_FIXED:.4f} (FIXED at 10%)")

    return {
        "sigma1": sigma1_mean,
        "sigma2": sigma2_mean,
        "mu_CL": mu_CL,
        "tau_CL": tau_CL,
        "sigma_error": SIGMA_ERROR_FIXED,
    }


def sample_parameters(results, method="dpinn", n_samples=N_SAMPLES):
    """
    Sample parameters for Monte Carlo simulations

    Args:
        results: Dictionary from extract_dpinn_results() or extract_stan_results()
        method: "dpinn" or "stan"
        n_samples: Number of Monte Carlo samples

    Returns:
        dict: Arrays of sampled parameters
    """

    print(f"\nSampling {n_samples} {method.upper()} parameter sets...")

    np.random.seed(42 if method == "dpinn" else 43)  # Different seeds for each method

    if method == "dpinn":
        # Sample CLp from lognormal
        CLp_samples = np.random.lognormal(
            mean=results["CLp_mean"], sigma=results["CLp_sd"], size=n_samples
        )
    else:  # stan
        # Sample CLp from lognormal using mu_CL and tau_CL
        CLp_samples = np.random.lognormal(
            mean=results["mu_CL"], sigma=results["tau_CL"], size=n_samples
        )

    # Fixed reflection coefficients
    sigma1_values = np.full(n_samples, results["sigma1"])
    sigma2_values = np.full(n_samples, results["sigma2"])

    print(f"  σ1: {sigma1_values[0]:.4f} (fixed)")
    print(f"  σ2: {sigma2_values[0]:.4f} (fixed)")
    print(f"  CLp: {CLp_samples.mean():.6f} ± {CLp_samples.std():.6f} L/h (sampled)")
    print(f"  BW: Using {len(BW_SAMPLES_FIXED)} pre-generated fixed samples")
    print(f"  σ_error: {SIGMA_ERROR_FIXED:.4f} (fixed)")

    return {
        "sigma1": sigma1_values,
        "sigma2": sigma2_values,
        "CLp": CLp_samples,
        "sigma_error": SIGMA_ERROR_FIXED,
    }


def mPBPK_odes(t, y, sigma1, sigma2, CLp, dose_mg, Vp, Vtight, Vleaky, L_total, Vlymph):
    """
    mPBPK ODE system for numerical integration

    Returns:
        dydt: Derivatives [dCp/dt, dCtight/dt, dCleaky/dt, dClymph/dt]
    """

    Cp, Ctight, Cleaky, Clymph = y

    L1_h = 0.33 * L_total
    L2_h = 0.67 * L_total
    sigma_L = MPBPK_CONSTANTS["sigma_L"]

    infusion_duration = 1.5
    infusion_rate = dose_mg / infusion_duration if t <= infusion_duration else 0.0

    dCp_dt = (
        infusion_rate
        + Clymph * L_total
        - Cp * L1_h * (1 - sigma1)
        - Cp * L2_h * (1 - sigma2)
        - CLp * Cp
    ) / Vp

    dCtight_dt = (L1_h * (1 - sigma1) * Cp - L1_h * (1 - sigma_L) * Ctight) / Vtight

    dCleaky_dt = (L2_h * (1 - sigma2) * Cp - L2_h * (1 - sigma_L) * Cleaky) / Vleaky

    dClymph_dt = (
        L1_h * (1 - sigma_L) * Ctight + L2_h * (1 - sigma_L) * Cleaky - Clymph * L_total
    ) / Vlymph

    return [dCp_dt, dCtight_dt, dCleaky_dt, dClymph_dt]


def simulate_at_data_points(param_samples, time_points, method="dpinn"):
    """
    Run Monte Carlo simulations and evaluate ONLY at experimental time points

    Args:
        param_samples: Dictionary of parameter arrays
        time_points: Experimental time points to evaluate at
        method: "dpinn" or "stan" for labeling

    Returns:
        numpy.ndarray: Predictions [n_samples × n_timepoints]
    """

    n_samples = len(param_samples["sigma1"])
    n_times = len(time_points)
    predictions = np.zeros((n_samples, n_times))

    print(f"\nRunning {n_samples} {method.upper()} Monte Carlo simulations...")
    print(f"Evaluating at {n_times} experimental time points")

    sigma_error_value = param_samples["sigma_error"]
    error_matrix = np.random.normal(
        loc=0.0, scale=sigma_error_value, size=(n_samples, n_times)
    )

    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            print(f"  Simulation {i+1}/{n_samples}")

        sigma1_i = param_samples["sigma1"][i]
        sigma2_i = param_samples["sigma2"][i]
        CLp_i = param_samples["CLp"][i]

        # Use pre-generated fixed BW sample
        BW_i = MPBPK_CONSTANTS["BW"][i]

        # Calculate volumes based on BW
        Vblood_fraction = 0.0733
        Vblood_i = BW_i * Vblood_fraction
        Vlymph_i = Vblood_i
        hematocrit = 0.43
        Vp_i = Vblood_i * (1 - hematocrit)

        ISF_fraction = 0.15
        ISF_total_i = BW_i * ISF_fraction * MPBPK_CONSTANTS["Kp"]
        Vtight_i = 0.65 * ISF_total_i
        Vleaky_i = 0.35 * ISF_total_i

        L_total_i = MPBPK_CONSTANTS["L"] / 24.0

        dose_mg = BW_i * 1.0

        y0 = [0.0, 0.0, 0.0, 0.0]

        try:
            solution = solve_ivp(
                fun=lambda t, y: mPBPK_odes(
                    t,
                    y,
                    sigma1_i,
                    sigma2_i,
                    CLp_i,
                    dose_mg,
                    Vp_i,
                    Vtight_i,
                    Vleaky_i,
                    L_total_i,
                    Vlymph_i,
                ),
                t_span=[0, time_points[-1]],
                y0=y0,
                t_eval=time_points,  # Evaluate only at experimental points
                method="LSODA",
                rtol=1e-8,
                atol=1e-10,
            )

            if solution.success:
                C_plasma = solution.y[0, :]
                C_with_error = C_plasma * np.exp(error_matrix[i, :])
                predictions[i, :] = C_with_error
            else:
                predictions[i, :] = np.nan

        except Exception:
            predictions[i, :] = np.nan

    failed_sims = np.sum(np.isnan(predictions).any(axis=1))
    if failed_sims > 0:
        print(f"Warning: {failed_sims} simulations failed")

    print(f"{method.upper()} Monte Carlo simulations completed")
    return predictions


def calculate_prediction_intervals(predictions):
    """
    Calculate 95% prediction intervals and median

    Args:
        predictions: Array [n_samples × n_timepoints]

    Returns:
        dict: Intervals and median
    """

    valid_predictions = predictions[~np.isnan(predictions).any(axis=1)]
    n_valid = valid_predictions.shape[0]

    print(f"Using {n_valid} valid simulations for intervals")

    ci_lower = np.percentile(valid_predictions, 2.5, axis=0)
    ci_upper = np.percentile(valid_predictions, 97.5, axis=0)
    median = np.percentile(valid_predictions, 50, axis=0)

    return {"ci_lower": ci_lower, "ci_upper": ci_upper, "median": median}


def load_observed_data():
    """
    Load observed mPBPK data

    Returns:
        pandas.DataFrame: Observed data
    """

    data_file = "mPBPK/Data/preprocessed_mPBPK_data.csv"

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Observed data file not found: {data_file}")

    print(f"\nLoading observed data from: {data_file}")

    observed_data = pd.read_csv(data_file)
    # Remove last row as in training script
    observed_data = observed_data[:-1]

    print(f"Loaded {len(observed_data)} observations")

    return observed_data


def create_comparison_plot(time_points, dpinn_intervals, stan_intervals, observed_data):
    """
    Create overlapped comparison plot with both D-PINN and Stan results

    Args:
        time_points: Time points for evaluation
        dpinn_intervals: D-PINN prediction intervals
        stan_intervals: Stan prediction intervals
        observed_data: Observed data
    """

    print("\nCreating comparison plot...")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Stan prediction interval (plot first, background)
    ax.fill_between(
        time_points,
        stan_intervals["ci_lower"],
        stan_intervals["ci_upper"],
        alpha=0.25,
        color="tab:orange",
        label="Stan 95% PI",
    )
    ax.plot(
        time_points,
        stan_intervals["median"],
        color="tab:orange",
        linestyle="--",
        lw=2.5,
        label="Stan Median",
    )

    # D-PINN prediction interval (plot second, foreground)
    ax.fill_between(
        time_points,
        dpinn_intervals["ci_lower"],
        dpinn_intervals["ci_upper"],
        alpha=0.3,
        color="tab:blue",
        label="D-PINN 95% PI",
    )
    ax.plot(
        time_points,
        dpinn_intervals["median"],
        color="tab:blue",
        linestyle="-",
        lw=2.5,
        label="D-PINN Median",
    )

    # Observed data points with error bars
    # Ensure error bars don't go below zero (concentrations cannot be negative)
    mean_values = observed_data["MEAN"].values
    sd_values = observed_data["SD"].values

    # Calculate asymmetric error bars to prevent negative values
    lower_errors = np.minimum(sd_values, mean_values)  # Don't go below zero
    upper_errors = sd_values  # Upper error is unchanged

    ax.errorbar(
        observed_data["TIME"],
        mean_values,
        yerr=[lower_errors, upper_errors],  # Asymmetric error bars
        fmt="o",
        color="black",
        ecolor="black",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=8,
        capsize=5,
        capthick=1.5,
        elinewidth=1.5,
        alpha=0.8,
        label="Observed Data (Mean ± SD)",
        zorder=10,
    )

    # Labels and formatting
    ax.set_xlabel("Time (hours)", fontsize=16)
    ax.set_ylabel("Plasma Concentration (mg/L)", fontsize=16)
    # No title
    ax.set_xlim(0, time_points.max() * 1.05)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
    ax.tick_params(axis="both", labelsize=14)

    # Legend below the plot
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=3,
    )

    plt.tight_layout()
    # plt.yscale("log")

    # Save figure
    output_file = "mPBPK/results/comparison_dpinn_stan_fixed_error.png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to: {output_file}")

    # plt.show()


def main():
    """
    Main comparison workflow
    """

    try:
        # Step 1: Extract D-PINN parameters
        print("Step 1: Extracting D-PINN parameter estimates...")
        dpinn_param_file = "mPBPK/results/mPBPK_dpinn_parameters.dat"
        dpinn_results = extract_dpinn_results(dpinn_param_file)

        # Step 2: Extract Stan parameters
        print("\nStep 2: Extracting Stan parameter estimates...")
        stan_file = "mPBPK/Data/stan_results_figure.csv"
        stan_results = extract_stan_results(stan_file)

        # Step 3: Load observed data (to get time points)
        print("\nStep 3: Loading observed data...")
        observed_data = load_observed_data()
        time_points = observed_data["TIME"].values

        # Step 4: Sample D-PINN parameters
        print("\nStep 4: Sampling D-PINN parameters...")
        dpinn_samples = sample_parameters(
            dpinn_results, method="dpinn", n_samples=N_SAMPLES
        )

        # Step 5: Sample Stan parameters
        print("\nStep 5: Sampling Stan parameters...")
        stan_samples = sample_parameters(
            stan_results, method="stan", n_samples=N_SAMPLES
        )

        # Step 6: Run D-PINN simulations
        print("\nStep 6: Running D-PINN simulations...")
        dpinn_predictions = simulate_at_data_points(
            dpinn_samples, time_points, method="dpinn"
        )

        # Step 7: Run Stan simulations
        print("\nStep 7: Running Stan simulations...")
        stan_predictions = simulate_at_data_points(
            stan_samples, time_points, method="stan"
        )

        # Step 8: Calculate D-PINN intervals
        print("\nStep 8: Calculating D-PINN prediction intervals...")
        dpinn_intervals = calculate_prediction_intervals(dpinn_predictions)

        # Step 9: Calculate Stan intervals
        print("\nStep 9: Calculating Stan prediction intervals...")
        stan_intervals = calculate_prediction_intervals(stan_predictions)

        # Step 10: Create comparison plot
        print("\nStep 10: Creating comparison plot...")
        create_comparison_plot(
            time_points, dpinn_intervals, stan_intervals, observed_data
        )

        print("\n" + "=" * 60)
        print("COMPARISON PLOT GENERATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError in comparison plot generation: {e}")
        raise


if __name__ == "__main__":
    main()

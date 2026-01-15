# D-PINNs for Pharmacokinetics

## Overview

This repository contains code for the publication "A Physics-Informed Neural Network Approach for Estimating Population-Level Pharmacokinetic Parameters from Aggregated Concentration Data". It implements Distributional Physics-Informed Neural Networks (D-PINNs) for learning population-level pharmacokinetic parameters from aggregated concentration data (mean and variance). The approach combines physics-based ODE constraints with neural networks to estimate parameter distributions without requiring individual-level data.

## Case Studies

### 1. One-Compartment Model (`one_compartment/`)

A benchmark case study using simulated data from a classical one-compartment pharmacokinetic model with first-order elimination.

- **Model**: Single compartment with IV bolus dosing (300 mg)
- **Parameters estimated**: Elimination rate constant (ke), volume of distribution (Vd), and their correlations
- **Data**: Synthetic data generated from 30 virtual patients with lognormal parameter distributions
- **Key files**:
  - `one_compartment_data_generation.py`: Generates synthetic PK data with population variability
  - `DPINNs_one_compartment.py`: D-PINN implementation for parameter estimation

### 2. Minimal PBPK Model (`mPBPK/`)

A physiologically-based pharmacokinetic model with multiple tissue compartments applied to real antibody concentration data.

- **Model**: Four-compartment mPBPK with plasma, tight tissues, leaky tissues, and lymph
- **Parameters estimated**: Reflection coefficients (σ1, σ2), plasma clearance (CLp), and correlation between concentration and clearance
- **Data**: Real preprocessed antibody concentration data (mean and standard deviation over time)
- **Key files**:
  - `DPINNS_mPBPK.py`: D-PINN implementation for mPBPK model
  - `Data/preprocessed_mPBPK_data.csv`: Preprocessed concentration data
  - `comparison_plot_dpinn_stan.py`: Comparison with Bayesian inference results

## Requirements

- Python 3.x
- DeepXDE
- TensorFlow
- TensorFlow Probability
- NumPy, Pandas, Matplotlib, SciPy

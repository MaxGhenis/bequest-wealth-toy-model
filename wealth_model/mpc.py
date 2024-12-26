# wealth_model/mpc.py
import numpy as np
from wealth_model.constants import MPC_BASE, MPC_REFERENCE_INCOME


def calculate_mpc(
    income, base_mpc=MPC_BASE, reference_income=MPC_REFERENCE_INCOME, elasticity=-0.7
):
    """Calculate marginal propensity to consume at a given income level"""
    scaled_income = np.maximum(income / reference_income, 1e-6)
    return base_mpc * np.power(scaled_income, elasticity)


def calculate_consumption(
    income, base_mpc=MPC_BASE, reference_income=MPC_REFERENCE_INCOME, elasticity=-0.7
):
    """Calculate consumption by integrating MPC from 0 to income"""
    alpha = elasticity + 1
    scaled_income = income / reference_income
    return base_mpc * reference_income * np.power(scaled_income, alpha) / alpha

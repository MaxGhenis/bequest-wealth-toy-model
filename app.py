# app.py
import streamlit as st
import numpy as np
from wealth_model.constants import *
from wealth_model.mpc import calculate_mpc, calculate_consumption
from wealth_model.plotting import (
    plot_mpc_consumption,
    plot_rank_relationships,
    plot_life_trajectory,
)
from wealth_model.simulation import run_simulation, calculate_rank_statistics


def main():
    st.title("Intergenerational Wealth Model")

    # Sidebar parameters
    st.sidebar.header("MPC Parameters")
    base_mpc = st.sidebar.slider("Base MPC", 0.1, 1.0, MPC_BASE)
    elasticity = st.sidebar.slider("MPC Elasticity", -2.0, 0.0, MPC_ELASTICITY)
    reference_income = st.sidebar.number_input(
        "Reference Income", 1000, 100000, MPC_REFERENCE_INCOME
    )

    mpc_params = {
        "base_mpc": base_mpc,
        "reference_income": reference_income,
        "elasticity": elasticity,
    }

    # Show MPC and consumption functions
    st.header("MPC and Consumption Functions")
    fig = plot_mpc_consumption(base_mpc, reference_income, elasticity)
    st.plotly_chart(fig, use_container_width=True)

    # Sample calculation
    st.header("Sample Calculation")
    income = st.number_input("Enter an income to see consumption", 1000, 1000000, 50000)

    mpc = calculate_mpc(income, base_mpc, reference_income, elasticity)
    consumption = calculate_consumption(income, base_mpc, reference_income, elasticity)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MPC", f"{mpc:.3f}")
    with col2:
        st.metric("Consumption", f"${consumption:,.0f}")
    with col3:
        st.metric("Saving Rate", f"{(1 - consumption/income):.1%}")

    # Run simulation
    st.header("Simulation Results")
    n_people = st.slider("Number of People", 1000, 50000, 10000)

    if st.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            children, bequest_ranks, wealth_ranks = run_simulation(
                n_people=n_people, mpc_params=mpc_params
            )

            bin_centers, mean_ranks, std_ranks = calculate_rank_statistics(
                bequest_ranks, wealth_ranks
            )

            # Plot rank relationships
            fig = plot_rank_relationships(bin_centers, mean_ranks, std_ranks)
            st.plotly_chart(fig, use_container_width=True)

            # Sample trajectories
            st.subheader("Sample Life Trajectories")
            percentiles = [25, 50, 75]
            for p in percentiles:
                idx = int(p / 100 * n_people)
                st.write(f"{p}th Percentile Parent Wealth")
                fig = plot_life_trajectory(children[idx])
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

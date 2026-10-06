# wealth_model/plotting.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wealth_model.mpc import calculate_mpc, calculate_consumption
from wealth_model.constants import CHILD_START_AGE


def plot_mpc_consumption(base_mpc, reference_income, elasticity):
    """Create interactive plot of MPC and consumption functions"""
    # Generate income points
    incomes = np.logspace(4, 7, 1000)  # $10k to $10M

    # Calculate MPCs and consumption
    mpcs = calculate_mpc(incomes, base_mpc, reference_income, elasticity)
    consumption = calculate_consumption(incomes, base_mpc, reference_income, elasticity)

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("MPC by Income Level", "Consumption Function")
    )

    # MPC plot
    fig.add_trace(
        go.Scatter(
            x=incomes,
            y=mpcs,
            name="MPC",
            hovertemplate="Income: $%{x:,.0f}<br>MPC: %{y:.3f}",
        ),
        row=1,
        col=1,
    )

    # Consumption plot
    fig.add_trace(
        go.Scatter(
            x=incomes,
            y=consumption,
            name="Consumption",
            hovertemplate="Income: $%{x:,.0f}<br>Consumption: $%{y:,.0f}",
        ),
        row=1,
        col=2,
    )

    # 45-degree line
    fig.add_trace(
        go.Scatter(
            x=incomes, y=incomes, name="45° line", line=dict(dash="dash", color="gray")
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig.update_xaxes(type="log", title="Income")
    fig.update_yaxes(title="MPC", row=1, col=1)
    fig.update_yaxes(type="log", title="Consumption", row=1, col=2)

    fig.update_layout(height=500, showlegend=True)

    return fig


def plot_rank_relationships(bin_centers, mean_ranks, std_ranks):
    """Plot rank-rank relationship and volatility"""
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Rank-Rank Relationship", "Rank Volatility")
    )

    # Split data into no-bequest and positive bequest sections
    no_bequest_mask = bin_centers < 0.1
    positive_mask = bin_centers >= 0.4

    # Rank-Rank plot: No bequest point
    if np.any(no_bequest_mask):
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[mean_ranks[no_bequest_mask][0]],
                mode="markers",
                name="No Bequest",
                marker=dict(size=10, color="blue"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Rank-Rank plot: Positive bequest line
    fig.add_trace(
        go.Scatter(
            x=bin_centers[positive_mask],
            y=mean_ranks[positive_mask],
            mode="lines+markers",
            name="Positive Bequest",
            line=dict(color="blue"),
            marker=dict(size=4),
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Perfect persistence line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Persistence",
            line=dict(dash="dash", color="gray"),
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Volatility plot: No bequest point
    if np.any(no_bequest_mask):
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[std_ranks[no_bequest_mask][0]],
                mode="markers",
                name="No Bequest Volatility",
                marker=dict(size=10, color="red"),
                showlegend=True,
            ),
            row=1,
            col=2,
        )

    # Volatility plot: Positive bequest line
    fig.add_trace(
        go.Scatter(
            x=bin_centers[positive_mask],
            y=std_ranks[positive_mask],
            mode="lines+markers",
            name="Bequest Volatility",
            line=dict(color="red"),
            marker=dict(size=4),
            showlegend=True,
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig.update_xaxes(title="Bequest Rank", range=[-0.05, 1.05], row=1, col=1)
    fig.update_xaxes(title="Bequest Rank", range=[-0.05, 1.05], row=1, col=2)
    fig.update_yaxes(title="Child Wealth Rank", range=[-0.05, 1.05], row=1, col=1)
    fig.update_yaxes(title="Std Dev of Child Ranks", row=1, col=2)

    fig.update_layout(
        height=500, showlegend=True, title="Intergenerational Wealth Mobility"
    )

    return fig


def plot_life_trajectory(person):
    """Plot wealth, income, and consumption over a person's life"""
    ages = np.arange(CHILD_START_AGE, CHILD_START_AGE + len(person.wealth_history))

    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("Wealth Trajectory", "Income and Consumption")
    )

    # Wealth plot
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=person.wealth_history,
            name="Wealth",
            hovertemplate="Age: %{x}<br>Wealth: $%{y:,.0f}",
        ),
        row=1,
        col=1,
    )

    # Income and consumption plot
    fig.add_trace(
        go.Scatter(
            x=ages[1:],
            y=[
                l + c
                for l, c in zip(
                    person.labor_income_history, person.capital_income_history
                )
            ],
            name="Total Income",
            hovertemplate="Age: %{x}<br>Income: $%{y:,.0f}",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ages[1:],
            y=person.consumption_history,
            name="Consumption",
            hovertemplate="Age: %{x}<br>Consumption: $%{y:,.0f}",
        ),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_xaxes(title="Age")
    fig.update_yaxes(title="Dollars")

    fig.update_layout(height=800, showlegend=True)

    return fig

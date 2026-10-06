# wealth_model/simulation.py
import numpy as np
from scipy.stats import rankdata, pareto
from wealth_model.constants import *
from wealth_model.mpc import calculate_consumption


class Person:
    def __init__(
        self,
        wealth_history,
        labor_income_history,
        capital_income_history,
        consumption_history,
    ):
        self.wealth_history = wealth_history
        self.labor_income_history = labor_income_history
        self.capital_income_history = capital_income_history
        self.consumption_history = consumption_history

        # Initialize histories
        self.labor_income_history = []
        self.capital_income_history = []
        self.consumption_history = []

        # Set base labor income based on parent wealth
        if parent_wealth_rank is not None:
            self.base_labor_income = BASE_INCOME * (
                1 + PARENT_WEALTH_EFFECT * parent_wealth_rank
            )
        else:
            self.base_labor_income = BASE_INCOME

    def get_labor_income(self, years_worked):
        """Calculate labor income with growth and volatility"""
        if self.age >= RETIREMENT_AGE:
            return self.base_labor_income * RETIREMENT_INCOME_RATIO

        growth_factor = (1 + INCOME_GROWTH_RATE) ** years_worked
        return (
            self.base_labor_income
            * growth_factor
            * np.random.lognormal(0, INCOME_VOLATILITY)
        )

    def get_capital_income(self):
        """Calculate capital income from wealth"""
        return_factor = 1 + np.random.normal(RETURN_MEAN, RETURN_VOLATILITY)
        return self.wealth * (return_factor - 1)

    def simulate_year(self, years_worked, mpc_params=None):
        """Simulate one year of income, consumption, and wealth accumulation"""
        if mpc_params is None:
            mpc_params = {
                "base_mpc": MPC_BASE,
                "reference_income": MPC_REFERENCE_INCOME,
                "elasticity": MPC_ELASTICITY,
            }

        # Calculate incomes
        labor_income = self.get_labor_income(years_worked)
        capital_income = self.get_capital_income()
        total_income = labor_income + capital_income

        # Calculate consumption by integrating MPC
        raw_consumption = calculate_consumption(
            total_income,
            base_mpc=mpc_params["base_mpc"],
            reference_income=mpc_params["reference_income"],
            elasticity=mpc_params["elasticity"],
        )
        # Can only borrow against positive wealth
        borrowing_available = np.where(
            self.wealth > 0, self.wealth * BORROWING_LIMIT_WEALTH_SHARE, 0
        )
        max_consumption = total_income + borrowing_available
        consumption = np.minimum(raw_consumption, max_consumption)

        # Update wealth
        self.wealth += total_income - consumption

        # Update histories
        self.wealth_history.append(self.wealth)
        self.labor_income_history.append(labor_income)
        self.capital_income_history.append(capital_income)
        self.consumption_history.append(consumption)

        self.age += 1


def run_simulation(n_people=10000, mpc_params=None):
    """Run full intergenerational simulation"""
    # Generate parent wealth distribution
    parent_wealth = pareto.rvs(PARETO_SHAPE, size=n_people) * WEALTH_SCALE * BASE_INCOME
    parent_ranks = rankdata(parent_wealth) / len(parent_wealth)

    # Initialize arrays for tracking
    n_years = 60
    wealth = np.zeros((n_years + 1, n_people))
    labor_income = np.zeros((n_years, n_people))
    capital_income = np.zeros((n_years, n_people))
    consumption = np.zeros((n_years, n_people))

    # Set initial conditions based on parent wealth
    base_labor_incomes = BASE_INCOME * (1 + PARENT_WEALTH_EFFECT * parent_ranks)

    # Generate bequest information
    will_get_bequest = parent_ranks >= NO_BEQUEST_PROB
    death_ages = np.random.randint(BEQUEST_AGE_MIN, BEQUEST_AGE_MAX + 1, size=n_people)
    bequest_shares = np.random.random(n_people)
    bequests = np.where(will_get_bequest, parent_wealth * bequest_shares, 0)

    # Simulate years
    for year in range(n_years):
        age = CHILD_START_AGE + year

        # Add bequests if parent dies this year
        bequest_mask = age == death_ages
        wealth[year] = np.where(bequest_mask, wealth[year] + bequests, wealth[year])

        # Labor income
        if age < RETIREMENT_AGE:
            growth_factor = (1 + INCOME_GROWTH_RATE) ** year
            labor_income[year] = (
                base_labor_incomes
                * growth_factor
                * np.random.lognormal(0, INCOME_VOLATILITY, n_people)
            )
        else:
            labor_income[year] = base_labor_incomes * RETIREMENT_INCOME_RATIO

        # Capital income
        return_factors = 1 + np.random.normal(RETURN_MEAN, RETURN_VOLATILITY, n_people)
        capital_income[year] = wealth[year] * (return_factors - 1)

        # Total income
        total_income = labor_income[year] + capital_income[year]

        # Consumption
        scaled_income = np.maximum(total_income / mpc_params["reference_income"], 1e-6)
        mpcs = mpc_params["base_mpc"] * np.power(
            scaled_income, mpc_params["elasticity"]
        )
        raw_consumption = mpcs * total_income

        # Apply borrowing constraints
        max_possible = wealth[year] + total_income
        min_consumption = BASE_INCOME * 0.2
        consumption[year] = np.clip(raw_consumption, min_consumption, max_possible)

        # Update wealth
        wealth[year + 1] = wealth[year] + total_income - consumption[year]

    # Calculate final rankings
    bequest_ranks = np.zeros(n_people)
    positive_bequest_mask = bequests > 0
    n_positive = np.sum(positive_bequest_mask)

    if n_positive > 0:
        positive_ranks = rankdata(bequests[positive_bequest_mask]) / n_positive
        bequest_ranks[positive_bequest_mask] = 0.4 + 0.6 * positive_ranks

    wealth_ranks = rankdata(wealth[-1]) / n_people

    return bequest_ranks, wealth_ranks


def calculate_rank_statistics(bequest_ranks, wealth_ranks):
    """Calculate mean and std of child ranks by bequest rank bin"""
    # Create bins
    bins = np.arange(0, 1.01, 0.01)  # 0 to 1 in steps of 0.01
    bin_centers = (bins[1:] + bins[:-1]) / 2

    # Initialize arrays
    mean_ranks = np.zeros(len(bins) - 1)
    std_ranks = np.zeros(len(bins) - 1)

    # Handle no-bequest cases (rank 0) separately
    no_bequest_mask = bequest_ranks < 0.1
    if np.any(no_bequest_mask):
        mean_ranks[0] = np.mean(wealth_ranks[no_bequest_mask])
        std_ranks[0] = np.std(wealth_ranks[no_bequest_mask])

    # Calculate statistics for positive bequests
    positive_mask = bequest_ranks >= 0.4
    if np.any(positive_mask):
        for i in range(1, len(bins) - 1):
            bin_mask = (bequest_ranks >= bins[i]) & (bequest_ranks < bins[i + 1])
            if np.sum(bin_mask) > 0:
                mean_ranks[i] = np.mean(wealth_ranks[bin_mask])
                std_ranks[i] = np.std(wealth_ranks[bin_mask])

    return bin_centers, mean_ranks, std_ranks

# wealth_model/simulation.py
import numpy as np
from scipy.stats import rankdata, pareto
from wealth_model.constants import *
from wealth_model.mpc import calculate_consumption


class Person:
    def __init__(self, age, initial_wealth, parent_wealth_rank=None):
        self.age = age
        self.wealth = initial_wealth
        self.parent_wealth_rank = parent_wealth_rank

        # Initialize histories
        self.wealth_history = [initial_wealth]
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

    # Create children (start with zero wealth)
    children = [Person(CHILD_START_AGE, 0, parent_rank) for parent_rank in parent_ranks]

    # Generate bequest timing
    will_get_bequest = parent_ranks >= NO_BEQUEST_PROB
    death_ages = np.random.randint(BEQUEST_AGE_MIN, BEQUEST_AGE_MAX + 1, size=n_people)
    bequest_shares = np.random.random(n_people)
    bequests = np.where(will_get_bequest, parent_wealth * bequest_shares, 0)

    # Simulate all children's lives
    n_years = 60
    for year in range(n_years):
        for i, child in enumerate(children):
            # Add bequest if parent dies this year
            if child.age == death_ages[i]:
                child.wealth += bequests[i]

            child.simulate_year(year, mpc_params)

    # Calculate final rankings
    bequest_ranks = np.zeros(n_people)
    positive_bequest_mask = bequests > 0
    n_positive = np.sum(positive_bequest_mask)
    if n_positive > 0:  # Handle case where no one gets bequest
        positive_ranks = rankdata(bequests[positive_bequest_mask]) / n_positive
        bequest_ranks[positive_bequest_mask] = 0.4 + 0.6 * positive_ranks

    final_wealth = np.array([child.wealth_history[-1] for child in children])
    wealth_ranks = (rankdata(final_wealth) - 1) / (n_people - 1)

    return children, bequest_ranks, wealth_ranks


def calculate_rank_statistics(bequest_ranks, wealth_ranks, n_bins=100):
    """Calculate mean and std of child ranks by bequest rank bin"""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[1:] + bins[:-1]) / 2

    mean_ranks = []
    std_ranks = []

    for i in range(n_bins):
        mask = (bequest_ranks >= bins[i]) & (bequest_ranks < bins[i + 1])
        if np.sum(mask) > 0:
            mean_ranks.append(np.mean(wealth_ranks[mask]))
            std_ranks.append(np.std(wealth_ranks[mask]))
        else:
            mean_ranks.append(np.nan)
            std_ranks.append(np.nan)

    return bin_centers, np.array(mean_ranks), np.array(std_ranks)

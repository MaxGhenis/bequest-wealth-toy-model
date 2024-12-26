# constants.py

# Income parameters
BASE_INCOME = 50000
INCOME_GROWTH_RATE = 0.02
INCOME_VOLATILITY = 0.3
PARENT_WEALTH_EFFECT = 0.2
RETIREMENT_INCOME_RATIO = 0.4

# Investment returns
RETURN_MEAN = 0.04
RETURN_VOLATILITY = 0.05

# MPC parameters
MPC_BASE = 0.6
MPC_REFERENCE_INCOME = 35000
MPC_ELASTICITY = -0.7
BORROWING_LIMIT_WEALTH_SHARE = 0.1

# Demographics
CHILD_START_AGE = 25
PARENT_START_AGE = 50
RETIREMENT_AGE = 65
BEQUEST_AGE_MIN = 30
BEQUEST_AGE_MAX = 60
NO_BEQUEST_PROB = 0.4

# Initial wealth
PARETO_SHAPE = 2
WEALTH_SCALE = 20
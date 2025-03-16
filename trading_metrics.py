# Description: This file contains functions to calculate trading metrics such as Sharpe Ratio, Sortino Ratio, and Max Drawdown.

import numpy as np
from scipy.stats import sem


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    return (returns.mean() - risk_free_rate) / returns.std()


def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    downside_returns = returns.copy()
    downside_returns[returns > 0] = 0
    expected_return = returns.mean()
    downside_std = downside_returns.std()
    return (expected_return - risk_free_rate) / downside_std


def calculate_max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    return drawdown.min()


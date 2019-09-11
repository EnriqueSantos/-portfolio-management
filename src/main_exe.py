# -*- coding: utf-8 -*-
from ironstocks_pipeline import *

if __name__ == '__main__':
    stack = stack_stocks(['NFLX','BABA','NVDA','PG'])
    returns_stocks = get_returns_stack(stack)
    visualize_returns(stack,returns_stocks)
    n_portfolios = 500
    means, stds = np.column_stack([random_portfolio(returns_stocks) for _ in range(n_portfolios)])
    visualize_portfolios(stds,means)
    weights, returns, risks = optimal_portfolio(returns_stocks)
    visualize_optimals(stds,means,risks,returns)
    optimal_weights = show_w(weights,stack.index.levels[0].tolist())
    print(optimal_weights)
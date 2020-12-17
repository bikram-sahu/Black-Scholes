## Assumptions

1. Lognormality of underlying asset The assumption of constant $\sigma$ and $\mu$ can be relaxed. The asset and time dependent volatility lead to complex solutions and may require numerical methods for solution.
2. Known constant risk free interest rate $r$
This assumption helps us in finding explicit solutions. This can be relaxed by making the interest rate time dependent but deterministic or by making them stochastic. We also assumed that equal borrowing and lending rate.
3. Historical volatility While pricing an option, the $\sigma$ value is taken from the historical data becuase the volatility in the future is unknow. This may lead to devivation of option price given by the BSM model and the market price.
4. No dividends
This can be relaxed by incorporating the dividend yield into the interest rate term.
5. Continuous delta hedging Delta is a function time since the option price is time dependent. We need to continuously change the number of shares we hold in order to eliminate the risk.
6. No transaction costs
The finite transaction cost will render the continuous hedging impossible forcing us to hedge the portfolio discretely.
7. No arbitrage opportunities There is no model dependent arbitrage! There could be other arbitrage opportunites in the real market but the model itself doesn't allow for the arbitrage.
8. Complete market
We assume that there is only one source of uncertainity i.e. the underying stock. Hence the position in the option, dependent on this underlying can be completely hedged by replicating the portfolio with the stock and cash.
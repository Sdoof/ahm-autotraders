# ahm-autotraders
Single-asset arbitrage, noise trading, and multi-market statistical arbitrage bots for AdHocMarkets

## Depdendencies

- numpy
- Nitin Yadav's Python API for Flexemarkets
- AdHocMarkets account

## Description

Input account and marketplace information into the following parameters in each file:

```python
# trading account details
FM_ACCOUNT = ''
FM_EMAIL = ''
FM_PASSWORD = ''
MARKETPLACE_ID = 0
```

This package contains 3 bots to be used with AdHocMarkets.

#### Single-asset arbitrage bot
Acquires underpriced assets from a private market and resells them at fair market value, maintaining optimal asset holdings. Only accepts trades that exceed a user-configurable profit margin, allowing the bot to account for slippage and costs.

#### Noise trading bot
Maintains optimal asset holdings while profiting from volatility in asset prices.

Each interval, the bot places a buy/sell order, depending on whether asset holdings are above or below the optimal level. A (60% default) confidence interval for the asset's price is generated using the Student t distribution, where μ is the [EMA](https://www.investopedia.com/terms/e/ema.asp) of price:

<img src="https://latex.codecogs.com/gif.latex?\mu_n=\lambda%20p_n+(1-\lambda)\mu_{n-i}"/> 

And σ is the rooted [EWMA](https://financetrain.com/calculate-historical-volatility-using-ewma/) of volatility:

<img src="https://latex.codecogs.com/gif.latex?\sigma_n=\sqrt{\frac{1}{m}\sum^m_{i=1}u^2_{n-i}}"/>

Assuming no shocks occur, this produces a reasonable estimate of the asset's price in the near future. The lower/higher value of the interval is then used as the price of the buy/sell order. If the asset level is already optimal, the buy/sell decision is chosen at random.

#### Multi-market statistical arbitrage bot
Progressively increases portfolio performance based on a given scoring function.

A default scoring function is implemented, which scores based on expected payoff, with a configurably weighted risk penalty that subtracts from the score based on the variance of the payoff.

Each interval, for each in-universe market, the bot discovers the buy/sell prices that will lead to a portfolio score increase should an order execute at that price. The bot then creates buy/sell orders for all discovered prices, replacing existing orders if the price has changed.

The bot serves the role of a market maker, publishing and maintaining prices for all assets across all markets, and benefitting if a trade is made at any of its given prices.

Additionally, should an arbitrage opportunity arise, the bot reprioritises the order queue in order to quickly respond to the opportunity, biasing its prices using a TWAP-based confidence interval in order to safely fill both sides of the arbitrage. Market-making activities are temporarily suspended when this happens.

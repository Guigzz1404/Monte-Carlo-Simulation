import yfinance as yf
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt


# Download data from Yahoo Finance
def get_data(stocks, start, end):
    df = yf.download(stocks, start, end, interval="1d", auto_adjust=True)["Close"]
    returns = df.pct_change().dropna()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix


# Variables Initialisation
stockList = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stockList, startDate, endDate)

# Random Weights
weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

# MONTE CARLO METHOD
# Number of Simulation
mc_sims = 100
T = 100 # timeframe in days (we'll calculate statistics 100 days in the future)

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T
portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
initialPortfolio = 10000

# MC loops
for m in range(0, mc_sims):
    # Generate a raw random base of shocks (innovations) ~ N(0,1)
    # for 100 days and all assets (before applying covariance & mean)
    Z = np.random.normal(size=(T, len(weights)))
    # Cholesky decomposition of the covariance matrix: produces a lower-triangular matrix L
    # used to transform independent N(0,1) shocks into correlated returns consistent with covMatrix
    L = np.linalg.cholesky(covMatrix)
    # Simulated daily returns = expected returns (meanM) + correlated shocks (L @ Z)
    dailyReturns = meanM + np.inner(L, Z)
    # Compute portfolio value path for simulation m: cumulative product of weighted daily returns applied to initialPortfolio
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio

plt.plot(portfolio_sims)
plt.ylabel("Portfolio Value ($)")
plt.xlabel("Days")
plt.title("MC Simulation of a Stock Portfolio")
plt.show()

# INTERPRETATION
# Extract the final portfolio values from all simulations (last day, index -1)
final_values = portfolio_sims[-1, :]

# Expected portfolio value (mean of the distribution)
expected_value = np.mean(final_values)
# Volatility (standard deviation of the final portfolio values)
volatility = np.std(final_values)
# 5% Value-at-Risk (the threshold below which the worst 5% outcomes fall)
VaR_5 = np.percentile(final_values, 5)
# 5% Conditional VaR (Expected Shortfall): average of the worst 5% scenarios
CVaR_5 = final_values[final_values <= VaR_5].mean()

print(f"Expected portfolio value (mean): {expected_value:,.2f}")
print(f"Volatility (std): {volatility:,.2f}")
print(f"5% Value-at-Risk: {VaR_5:,.2f}")
print(f"5% Conditional VaR (Expected Shortfall): {CVaR_5:,.2f}")


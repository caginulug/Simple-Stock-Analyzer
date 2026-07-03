import yfinance as yf
import numpy as np
import pandas as pd

# ==================== Lets the user choose the historical data period ====================
print("\nSelect time period:")
print("1 - 1 Year")
print("2 - 2 Years")
print("3 - 5 Years")
print("4 - 10 Years")
print("5 - Maximum Available")

periods = {
    "1": "1y",
    "2": "2y",
    "3": "5y",
    "4": "10y",
    "5": "max"
}

while True:
    period_choice = input("Choice: ")
    period = periods.get(period_choice)

    if period is not None:
        break

    print("Invalid period selection. Please try again.")

# ==================== Lets the user enter the stock ticker ====================
while True:
    ticker = input("Please Enter Ticker Code: ").upper()

    test = yf.download(ticker, period="5d", auto_adjust=True, progress=False)

    if not test.empty:
        break

    print("Invalid ticker symbol. Please enter a valid ticker symbol.")

# ==================== Lets the user choose market index ticker  ====================
print("\nSelect Market Index:")
print("1 - S&P 500")
print("2 - NASDAQ 100")
print("3 - Dow Jones")
print("4 - BIST 100")
print("5 - Custom")

index = {
    "1": "^GSPC",
    "2": "^NDX",
    "3": "^DJI",
    "4": "XU100.IS"
}

while True:
    index_choice = input("Choice: ")

    if index_choice == "5":
        while True:
            market_ticker = input("Please Enter Market Index Ticker: ").upper()
            test2 = yf.download(market_ticker, period="5d", auto_adjust=True, progress=False)
            
            if not test2.empty:
                break 
            
            print("Invalid Market Index Ticker. Please try again.") 
        
        break 

    market_ticker = index.get(index_choice)
    if market_ticker is not None:
        break

    print("Invalid Market Index selection. Please try again.")

# ==================== The part that retrieves data ====================
a = yf.download([ticker, market_ticker], period=period, auto_adjust=True)["Close"]

if a.empty:
    raise ValueError("No data downloaded. Check the ticker symbols.")

# ==================== Calculates the log returns of the selected stocks ====================
log_return = np.log(a / a.shift(1)).dropna()

stock_ret = log_return[ticker]
mkt_ret = log_return[market_ticker]

# ==================== The stock and market returns are combined into one dataframe ====================
# ==================== Aligns stock and market returns on the same time level ====================
data = pd.concat([stock_ret, mkt_ret], axis=1, join="inner").dropna()
data.columns = ["stock", "market"]

#==================== Gets the annual risk-free rate from the user ====================
while True:
    try:
        rf_input = input("Risk-free rate (e.g. 4, %4, 0.04): ")

        rf_input = rf_input.replace("%","").strip()

        rf = float(rf_input)

        if rf > 1:
            rf /= 100

        break

    except ValueError:
        print("Invalid value try a valid value.")

# ==================== Price list for maximum drawdown calculation ====================
stock_prices = a[ticker].dropna()

# ==================== Calculates the historical annualized return ====================
historical_return = np.exp(data["stock"].mean() * 252) - 1

# ==================== Finds the annualized volatility of the stock ====================
volatility = data["stock"].std() * np.sqrt(252)

# ==================== Calculates beta as covariance(stock, market) / variance(market) ====================
cov = data["stock"].cov(data["market"])
var = data["market"].var()
if var != 0:
    beta = cov / var
else:
    beta = np.nan

# ==================== Calculates the annualized market return ====================
market_historical_return = np.exp(data["market"].mean() * 252) - 1

# ==================== Expected return according to the CAPM ====================
capm_return = rf + beta * (market_historical_return-rf)

# ==================== Correlation between the stock and the market ====================
correlation = data["stock"].corr(data["market"])

# ==================== Calculates Jensen's Alpha ====================
alpha = historical_return - capm_return

# ==================== Calculates the Sharpe Ratio ====================
if volatility != 0:
    sharpe_ratio = (historical_return-rf)/volatility
else:
    sharpe_ratio = np.nan

# ==================== Calculates the Maximum Drawdown ====================
cum_max = stock_prices.cummax()
drawdown = (stock_prices - cum_max) / cum_max
max_drawdown = drawdown.min()

# ==================== Calculates the Treynor Ratio ====================
if beta != 0:
    treynor_ratio = (historical_return - rf) / beta
else:
    treynor_ratio = np.nan

# ==================== Measures distribution asymmetry (skewness) and tail risk (kurtosis) ====================
skewness = data["stock"].skew()
kurtosis = data["stock"].kurt()

# ==================== Output part ====================
print("\n========== RESULTS ==========")
print(f"Historical Annualized Return : {historical_return}")
print(f"CAPM Expected Return         : {capm_return}")
print(f"Jensen's Alpha               : {alpha}")
print(f"Beta                         : {beta}")
print(f"Correlation                  : {correlation}")
print(f"Volatility                   : {volatility}")
print(f"Sharpe Ratio                 : {sharpe_ratio}")
print(f"Treynor Ratio                : {treynor_ratio}")
print(f"Maximum Drawdown             : {max_drawdown}")
print(f"Skewness                     : {skewness}")
print(f"Kurtosis                     : {kurtosis}")
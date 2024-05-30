# filename: plot_stocks_ytd.py
import yfinance as yf
import matplotlib.pyplot as plt
import datetime

# Define the stock ticker symbols
tickers = ['NVDA', 'TSLA']

# Get the current date
end_date = datetime.datetime.now().strftime('%Y-%m-%d')
# Get the start date as the beginning of the current year
start_date = datetime.datetime(datetime.datetime.now().year, 1, 1).strftime('%Y-%m-%d')

# Download stock data
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate percentage change from the beginning of the year
data_pct_change = data.pct_change().apply(lambda x: (1 + x).cumprod() - 1)

# Plot the data
plt.figure(figsize=(14, 7))
for ticker in tickers:
    plt.plot(data_pct_change.index, data_pct_change[ticker], label=ticker)

plt.title('YTD Stock Price Change: NVDA and TSLA')
plt.xlabel('Date')
plt.ylabel('Percentage Change')
plt.legend()
plt.grid(True)
plt.show()
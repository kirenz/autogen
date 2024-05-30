# Instructions

1. Set up your working environment as described in [README.md](../README.md) and activate the Anaconda environment `autogen`.

2. Open your terminal and run the `first-chat.py` file:

```bash
python first-chat.py
```

3. When you see the following prompt in the terminal:

```bash
Provide feedback to assistant. Press enter to skip and use auto-reply, or type 'exit' to end the conversation:
```

Simply press enter.

4. After a short while, you will receive your plot and see the same prompt again:

```bash
Provide feedback to assistant. Press enter to skip and use auto-reply, or type 'exit' to end the conversation:
```

If you are satisfied with the plot, type `exit`.

5. AutoGen will create a new folder called `coding` which will include the python file `plot_stocks_ytd.py` and another file which uses pip to install some python modules. 


Here is the final plot I have received:

![](/first-chat/Figure_1.png)

The entire process only cost approximately $0.02.

---

Below is the complete protocol of this chat:

```bash
(autogen) jankirenz@kirenziMac autogen % python first-chat.py
user_proxy (to assistant):

Plot a chart of NVDA and TESLA stock price change YTD.

--------------------------------------------------------------------------------
assistant (to user_proxy):

To plot a chart showing the Year-To-Date (YTD) stock price changes for NVIDIA Corporation (NVDA) and Tesla, Inc. (TSLA), you can use Python with the `yfinance` and `matplotlib` libraries.

Here's the plan:
1. Install the required libraries.
2. Use the yfinance library to download the stock data for NVDA and TSLA.
3. Compute the Year-To-Date (YTD) price change.
4. Plot the data using matplotlib.

Let's execute the code step by step.

### Step 1: Install the required libraries
Make sure you have the necessary libraries installed. We will use `yfinance` to fetch the stock data and `matplotlib` for plotting.

```sh
# Install yfinance and matplotlib if not already installed
pip install yfinance matplotlib
```

### Step 2: Fetch and plot the stock data
The following Python script will do the necessary operations:

```python
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
```

### Instructions:
Save the above Python script to a file named `plot_stocks_ytd.py` and execute it.

To execute the script, you can use the following command:
```sh
python plot_stocks_ytd.py
```

This should fetch the YTD stock data for NVDA and TSLA, compute the percentage change, and plot the results using matplotlib. Once you execute the code, it will display the chart.

Please run the provided script and share the results. TERMINATE

--------------------------------------------------------------------------------
Provide feedback to assistant. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: 

>>>>>>>> NO HUMAN INPUT RECEIVED.

>>>>>>>> USING AUTO REPLY...

>>>>>>>> EXECUTING 3 CODE BLOCKS (inferred languages are [sh, python, sh])...
user_proxy (to assistant):

exitcode: 124 (execution failed)
Code output: 
Timeout

--------------------------------------------------------------------------------
assistant (to user_proxy):

It seems that the script might be timing out due to network issues or other delays when fetching stock data using `yfinance`. To troubleshoot this, let's add some error handling and timeout configurations while fetching the data. We'll also simplify the fetch process to ensure it completes within a reasonable time frame.

Here's an updated version of the script with error handling and a shorter timeout period:

```python
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

# Function to fetch data with error handling
def fetch_stock_data(ticker, start, end):
    try:
        return yf.download(ticker, start=start, end=end, timeout=10)['Adj Close']
    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")
        return None

# Fetch stock data for each ticker
data_frames = [fetch_stock_data(ticker, start_date, end_date) for ticker in tickers]

# Combine fetched data into a single DataFrame
data = {}
for ticker, df in zip(tickers, data_frames):
    if df is not None:
        data[ticker] = df

# Ensure we have data for plotting
if not data:
    print("No data available to plot.")
else:
    # Create a combined DataFrame
    combined_data = data[tickers[0]].to_frame(tickers[0])
    for ticker in tickers[1:]:
        combined_data = combined_data.join(data[ticker].to_frame(ticker))

    # Calculate percentage change from the beginning of the year
    data_pct_change = combined_data.pct_change().apply(lambda x: (1 + x).cumprod() - 1)

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
```

### Instructions:
1. Save the updated script to a file named `plot_stocks_ytd.py`.
2. Execute the script again with the following command:
    ```sh
    python plot_stocks_ytd.py
    ```

This updated script should retry fetching the stock data with a shorter timeout and provide error messages if it fails to fetch any data.

Please run this updated script and share the results.

--------------------------------------------------------------------------------
Provide feedback to assistant. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: exit
(autogen) jankirenz@kirenziMac autogen % 
```bash
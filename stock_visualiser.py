import yfinance as yf
import matplotlib.pyplot as plt

def fetch_and_plot_stock_data(ticker, start_date, end_date):
    # Fetch the stock data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Check if data is fetched successfully
    if data.empty:
        print("No data found for the given ticker and date range.")
        return

    # Calculate moving averages
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # Plotting
    plt.figure(figsize=(14, 7))
    
    # Plot close price
    plt.subplot(2, 1, 1)
    plt.plot(data['Close'], label='Closing Price', color='blue')
    plt.plot(data['MA50'], label='50-Day MA', color='orange')
    plt.plot(data['MA200'], label='200-Day MA', color='green')
    plt.title(f'{ticker} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Plot volume
    plt.subplot(2, 1, 2)
    plt.bar(data.index, data['Volume'], color='gray')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.title(f'{ticker} Trading Volume')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Get user input for the stock ticker and date range
ticker = input("Enter the stock ticker symbol (e.g., AAPL): ")
start_date = input("Enter the start date (YYYY-MM-DD): ")
end_date = input("Enter the end date (YYYY-MM-DD): ")

# Fetch and plot the stock data
fetch_and_plot_stock_data(ticker, start_date, end_date)


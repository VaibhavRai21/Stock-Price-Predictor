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

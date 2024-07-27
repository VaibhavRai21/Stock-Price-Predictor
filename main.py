import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
import flet as ft

def fetch_and_predict(ticker, start_date, end_date, future_days):
    # Download the stock data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Check if data is empty
    if data.empty:
        return "Error: No data retrieved for the specified dates."

    # Use 'Close' prices for prediction
    data = data[['Close']]
    data['Prediction'] = data['Close'].shift(-future_days)

    # Drop rows with NaN values in 'Prediction'
    data.dropna(inplace=True)

    # Check if there is enough data after dropping NaN values
    if len(data) == 0:
        return "Error: Not enough data after processing."

    # Features and target
    X = data[['Close']].values
    y = data['Prediction'].values

    # Check if there are enough samples to split
    if len(X) <= 1:
        return "Error: Not enough data to create train and test sets."

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = np.mean((predictions - y_test) ** 2)

    # Predict future stock prices
    last_day = data[['Close']].values[-1]
    future_predictions = []

    for _ in range(future_days):
        prediction = model.predict(last_day.reshape(1, -1))
        future_predictions.append(prediction[0])
        last_day = prediction

    # Plot and save the historical prices and predictions
    plt.figure(figsize=(24, 5))

    # Historical Prices and Predictions
    plt.subplot(1, 2, 1)
    plt.plot(range(len(data['Close'])), data['Close'], label='Historical Prices')
    plt.plot(range(len(data['Close']), len(data['Close']) + future_days), future_predictions, label='Future Predictions')
    plt.legend()
    plt.title("Historical Prices and Future Predictions")

    # Actual vs Predicted Prices
    plt.subplot(1, 2, 2)
    plt.plot(y_test, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.legend()
    plt.title("Actual vs Predicted Prices")

    plt.savefig("Result.png")
    plt.close()

    return mse

def main(page: ft.Page):
    page.title = "Stock Price Predictor"

    def on_predict_click(e):
        ticker = ticker_input.value
        start_date = start_date_input.value
        end_date = end_date_input.value
        future_days_str = future_days_input.value

        # Validate future_days input
        if not future_days_str or not future_days_str.isdigit():
            page.add(ft.Text("Error: 'Future Days' must be a valid integer.", color="red"))
            return

        future_days = int(future_days_str)

        result = fetch_and_predict(ticker, start_date, end_date, future_days)
        
        if isinstance(result, str) and result.startswith("Error"):
            page.add(ft.Text(result, color="red"))
            return
        
        mse = result
        page.add(ft.Text(f"Mean Squared Error: {mse}"))

        # Add images and texts for plots if selected
        if show_hist_plot.value or show_actual_vs_predicted_plot.value:
            page.add(ft.Text("Historical Prices, Predictions and Actual vs Predicted"))
            try:
                page.add(ft.Image(src="combined_plot.png"))
            except Exception as e:
                page.add(ft.Text(f"Error loading combined plot image: {e}", color="red"))

    ticker_input = ft.TextField(label="Stock Ticker", value="")
    start_date_input = ft.TextField(label="Start Date (YYYY-MM-DD)", value="2021-01-01")
    end_date_input = ft.TextField(label="End Date (YYYY-MM-DD)", value="2023-01-01")
    future_days_input = ft.TextField(label="Future Days", value="40")
    
    show_hist_plot = ft.Checkbox(label="Historical Prices and Predictions", value=True)
    show_actual_vs_predicted_plot = ft.Checkbox(label="Actual vs Predicted Prices", value=True)
    
    predict_button = ft.ElevatedButton(text="Predict", on_click=on_predict_click)

    page.add(ticker_input)
    page.add(start_date_input)
    page.add(end_date_input)
    page.add(future_days_input)
    page.add(show_hist_plot)
    page.add(show_actual_vs_predicted_plot)
    page.add(predict_button)

ft.app(target=main)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
import flet as ft


def fetch_and_predict(ticker, start_date, end_date, future_days, model_type):
    # Download the stock data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Use 'Close' prices for prediction
    data = data.loc[:, ['Close']].copy()
    data['Prediction'] = data['Close'].shift(-future_days)

    # Drop rows with NaN values in 'Prediction'
    data.dropna(inplace=True)

    # Features and target
    X = data[['Close']].values
    y = data['Prediction'].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Choose and train the model
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Random Forest":
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
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(data['Close'])), data['Close'], label='Historical Prices')
    plt.plot(range(len(data['Close']), len(data['Close']) + future_days), future_predictions,
             label='Future Predictions')
    plt.legend()
    plt.savefig("historical_prices.png")
    plt.close()

    # Plot and save the actual vs predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.legend()
    plt.savefig("actual_vs_predicted.png")
    plt.close()

    return mse


def main(page: ft.Page):
    # Create pages for navigation
    main_page = ft.Column()
    results_page = ft.Column()

    def on_predict_click(e):
        ticker = ticker_input.value
        start_date = start_date_input.value
        end_date = end_date_input.value
        future_days = int(future_days_input.value)
        model_type = model_selector.value

        mse = fetch_and_predict(ticker, start_date, end_date, future_days, model_type)
        main_page.controls.append(ft.Text(f"Mean Squared Error: {mse}"))

        if show_hist_plot.value:
            results_page.controls.append(ft.Text("Historical Prices and Predictions"))
            results_page.controls.append(ft.Image(src="historical_prices.png"))

        if show_actual_vs_predicted_plot.value:
            results_page.controls.append(ft.Text("Actual vs Predicted Prices"))
            results_page.controls.append(ft.Image(src="actual_vs_predicted.png"))

        # Switch to results page
        page.controls.append(ft.ElevatedButton(text="Back to Main", on_click=lambda e: page.controls.append(main_page)))
        page.controls.append(results_page)
        page.update()

    def on_back_to_main_click(e):
        page.controls.remove(results_page)
        page.controls.append(main_page)
        page.update()

    # Main page controls
    ticker_input = ft.TextField(label="Stock Ticker", value="AAPL")
    start_date_input = ft.TextField(label="Start Date (YYYY-MM-DD)", value="2015-01-01")
    end_date_input = ft.TextField(label="End Date (YYYY-MM-DD)", value="2023-01-01")
    future_days_input = ft.TextField(label="Future Days", value="30")

    model_selector = ft.Dropdown(
        label="Select Model",
        options=[
            ft.dropdown.Option("Linear Regression"),
            ft.dropdown.Option("Random Forest")
        ],
        value="Linear Regression"
    )

    show_hist_plot = ft.Checkbox(label="Show Historical Prices and Predictions", value=True)
    show_actual_vs_predicted_plot = ft.Checkbox(label="Show Actual vs Predicted Prices", value=True)

    predict_button = ft.ElevatedButton(text="Predict", on_click=on_predict_click)

    main_page.controls.extend([
        ticker_input,
        start_date_input,
        end_date_input,
        future_days_input,
        model_selector,
        show_hist_plot,
        show_actual_vs_predicted_plot,
        predict_button
    ])

    # App bar (menu)
    app_bar = ft.AppBar(
        title=ft.Text("Stock Price Prediction"),
        actions=[
            ft.IconButton(icon=ft.icons.HOME, on_click=on_back_to_main_click),
        ],
    )

    # Scrollable content
    scrollable_content = ft.Scrollable(
        content=main_page
    )

    # Initial page setup
    page.appbar = app_bar
    page.add(scrollable_content)


ft.app(target=main)

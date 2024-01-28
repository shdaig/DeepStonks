import pandas as pd
import numpy as np
import plotly.graph_objects as go

from stonks.preprocessing.indicator_generator import IndicatorGenerator
from sklearn.model_selection import ParameterGrid


def buy(current_cash: float,
        current_price: float,
        stocks_count: int,
        commission: float = 0.0005) -> tuple[float, float]:
    spends = current_price * stocks_count * (1 + commission)
    cash = current_cash - spends
    return cash, spends


def sell(current_cash: float,
         current_price: float,
         stocks_count: int,
         commission: float = 0.0005) -> tuple[float, float]:
    profit = current_price * stocks_count * (1 - commission)
    cash = current_cash + profit
    return cash, profit


if __name__ == "__main__":
    tickers_1h = ["MTLR"]

    for ticker in tickers_1h:
        candles_df = pd.read_csv(f"../../data/{ticker}_1h.csv")

        params = {"smooth_williams_r_window": [5],
                  "diff_threshold": [0.10]}
        indicator_generator = IndicatorGenerator()

        candles_dfs = {}
        for window in params["smooth_williams_r_window"]:
            candles_dfs[window] = indicator_generator.calculate_williams_percent_r(candles_df,
                                                                                   smooth_window=window,
                                                                                   period=14)

        param_grid = ParameterGrid(params)

        for param in param_grid:
            candles_df = candles_dfs[param["smooth_williams_r_window"]]
            diff_threshold = param["diff_threshold"]

            data_size = candles_df.shape[0]
            # separate_parts = data_size // 350 - 1
            separate_parts = 2

            scale = 10
            chunk_len = 350
            pad = 100
            # threshold_signal_count = 1
            # scale = 100
            # chunk_len = 30
            # pad = 20

            viz = True
            buy_stock = 1
            commission = 0.0005

            for i in range(separate_parts):
                start_idx = data_size - chunk_len * (i + 1) - pad
                end_idx = data_size - chunk_len * i
                chunk = candles_df.iloc[start_idx:end_idx].copy()

                print(f"[{start_idx} - {end_idx}]")

                cash = 1000000.0
                stocks_count = 0
                last_cash = cash
                buy_points = []
                sell_points = []
                bought = False

                williams_r = chunk["williams_r"].to_numpy()
                prev_state = "none"
                if williams_r[pad - 1] >= 0.8:
                    prev_state = "oversold"
                elif williams_r[pad - 1] <= 0.2:
                    prev_state = "overbought"
                # overbought (< 0.2), oversold (> 0.8), none
                for j in range(pad, pad + chunk_len):
                    current_state = "none"
                    if williams_r[j] >= 0.8:
                        current_state = "oversold"
                    elif williams_r[j] <= 0.2:
                        current_state = "overbought"

                    if current_state == "none":
                        if prev_state == "oversold":
                            if williams_r[j - 1] - williams_r[j] >= diff_threshold:
                                current_price = (chunk["Open"].iloc[j] + chunk["Close"].iloc[j]) / 2
                                cash, spends = buy(cash, current_price, buy_stock, commission=commission)
                                stocks_count += buy_stock
                                buy_points.append(j)
                                bought = True
                                k = 0
                        elif prev_state == "overbought":
                            if bought:
                                current_price = (chunk["Open"].iloc[j] + chunk["Close"].iloc[j]) / 2
                                cash, profit = sell(cash, current_price, stocks_count, commission=commission)
                                stocks_count = 0
                                last_cash = cash
                                sell_points.append(j)
                                bought = False
                                k = 0
                    prev_state = current_state

                print(last_cash)

                if viz:
                    fig = go.Figure()
                    fig.add_candlestick(open=chunk['Open'], high=chunk['High'],
                                        low=chunk['Low'], close=chunk['Close'],
                                        opacity=0.5, name="candles")
                    fig.add_scatter(x=buy_points,
                                    y=chunk['Close'].to_numpy()[buy_points],
                                    mode='markers',
                                    line=dict(color='green'),
                                    opacity=0.7,
                                    marker=dict(size=10, symbol="triangle-up"),
                                    name="buy")
                    fig.add_scatter(x=sell_points,
                                    y=chunk['Close'].to_numpy()[sell_points],
                                    mode='markers',
                                    line=dict(color='red'),
                                    opacity=0.7,
                                    marker=dict(size=10, symbol="triangle-down"),
                                    name="sell")
                    k = chunk['Close'].mean()
                    fig.add_scatter(y=np.full(chunk['williams_r'].shape, 0.8) * scale + k, mode='lines',
                                    line=dict(color='blue'), opacity=0.4,
                                    name="williams_r high threshold")
                    fig.add_scatter(y=np.full(chunk['williams_r'].shape, 0.2) * scale + k, mode='lines',
                                    line=dict(color='blue'), opacity=0.4,
                                    name="williams_r low threshold")
                    fig.add_scatter(y=chunk['williams_r'] * scale + k, mode='lines',
                                    line=dict(color='blue'), opacity=0.7,
                                    name="williams_r")
                    fig.update_layout(xaxis_rangeslider_visible=False)
                    fig.show()

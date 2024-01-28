import pandas as pd
import numpy as np


class IndicatorGenerator:
    def calculate_sma(self,
                      data_df: pd.DataFrame,
                      window_size: int = 9) -> pd.DataFrame:
        data = data_df['Close'].to_numpy()
        data = np.append(np.ones(window_size - 1), data)
        weights = np.repeat(1.0, window_size) / window_size
        sma = np.convolve(data, weights, 'valid')

        result_df = data_df.copy()
        result_df["sma"] = sma

        return result_df

    def calculate_macd(self,
                       data_df: pd.DataFrame,
                       short_window: int = 12,
                       long_window: int = 26,
                       signal_window: int = 9) -> pd.DataFrame:
        print(data_df.shape)
        data = data_df['Close'].copy()
        short_ema = data.ewm(span=short_window, adjust=False).mean()
        long_ema = data.ewm(span=long_window, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        histogram = macd_line - signal_line

        result_df = data_df.copy()
        result_df["macd_line"] = macd_line
        result_df["signal_line"] = signal_line
        result_df["macd_histogram"] = histogram

        return result_df

    def calculate_rsi(self,
                      data_df: pd.DataFrame,
                      window: int = 14) -> pd.DataFrame:
        prices = data_df['Close'].to_numpy()
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[:window])
        avg_loss = np.mean(losses[:window])
        rsi_values = np.zeros_like(prices)
        for i in range(window, len(prices)):
            delta = deltas[i - 1]
            avg_gain = (avg_gain * (window - 1) + (delta if delta > 0 else 0)) / window
            avg_loss = (avg_loss * (window - 1) + (-delta if delta < 0 else 0)) / window
            rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
            rsi_values[i] = 100 - (100 / (1 + rs))

        rsi_values /= 100

        result_df = data_df.copy()
        result_df["rsi"] = rsi_values

        return result_df

    def calculate_logarithmic_return(self, data_df: pd.DataFrame) -> pd.DataFrame:
        closing_prices = data_df['Close'].to_numpy()
        log_returns = np.log(closing_prices[1:] / closing_prices[:-1])

        log_returns = np.append(0, log_returns)

        log_returns = log_returns * 10

        result_df = data_df.copy()
        result_df["log_returns"] = log_returns

        return result_df

    def calculate_williams_percent_r(self, price_df: pd.DataFrame, smooth_window: int = 1, period: int = 14) -> pd.DataFrame:
        rolling_raw = price_df.rolling(period)
        windows = [window for window in rolling_raw]

        percent_r_all = []

        for window in windows:
            high_max = np.max(window["High"].to_numpy())
            low_min = np.min(window["Low"].to_numpy())
            percent_r = ((high_max - window["Close"].to_numpy()[-1]) / max((high_max - low_min), 1e-7))
            percent_r_all.append(percent_r)

        percent_r_all = np.array(percent_r_all)
        ema_percent_r_all = pd.Series(percent_r_all).ewm(span=smooth_window, adjust=False).mean()

        result_df = price_df.copy()
        result_df["williams_r"] = ema_percent_r_all

        return result_df


if __name__ == "__main__":
    import plotly.graph_objects as go

    candles_df = pd.read_csv("../../price_data/MTLR_1h.csv")

    indicator_generator = IndicatorGenerator()

    adjusted_candles_df = indicator_generator.calculate_sma(candles_df, window_size=9)
    adjusted_candles_df = indicator_generator.calculate_macd(adjusted_candles_df, short_window=12,
                                                             long_window=26, signal_window=9)
    adjusted_candles_df = indicator_generator.calculate_rsi(adjusted_candles_df, window=14)
    adjusted_candles_df = indicator_generator.calculate_logarithmic_return(adjusted_candles_df)
    adjusted_candles_df = indicator_generator.calculate_williams_percent_r(adjusted_candles_df,
                                                                           smooth_window=1, period=14)

    layout = go.Layout(
        xaxis=dict(
            range=[8440, 8600]
        ),
        yaxis=dict(
            range=[122, 134]
        )
    )
    fig = go.Figure(layout=layout)
    # price plot
    fig.add_candlestick(open=adjusted_candles_df['Open'], high=adjusted_candles_df['High'],
                        low=adjusted_candles_df['Low'], close=adjusted_candles_df['Close'],
                        opacity=0.5, name="candles")
    # sma
    fig.add_scatter(y=adjusted_candles_df['sma'], mode='lines',
                    line=dict(color='black'), opacity=0.5,
                    name="sma")
    # macd
    k = 128
    fig.add_scatter(y=adjusted_candles_df['macd_line'] + k, mode='lines',
                    line=dict(color='green'), opacity=0.3,
                    name="macd_line")
    fig.add_scatter(y=adjusted_candles_df['signal_line'] + k, mode='lines',
                    line=dict(color='red'), opacity=0.3,
                    name="signal_line")
    fig.add_scatter(y=adjusted_candles_df['macd_histogram'] + k, mode='lines',
                    line=dict(color='red'),
                    opacity=0.5,
                    name="macd_histogram")

    # rsi
    k = 130
    fig.add_scatter(y=adjusted_candles_df['rsi'] + k, mode='lines',
                    line=dict(color='green'), opacity=0.6,
                    name="rsi")


    # log_returns
    k = 126
    fig.add_scatter(y=adjusted_candles_df['log_returns'] + k, mode='lines',
                    line=dict(color='black'), opacity=0.6,
                    name="log_returns")

    # williams_r
    k = 124
    fig.add_scatter(y=np.full(adjusted_candles_df['williams_r'].shape, 0.8) + k, mode='lines',
                    line=dict(color='blue'), opacity=0.4,
                    name="williams_r high threshold")
    fig.add_scatter(y=np.full(adjusted_candles_df['williams_r'].shape, 0.2) + k, mode='lines',
                    line=dict(color='blue'), opacity=0.4,
                    name="williams_r low threshold")
    fig.add_scatter(y=adjusted_candles_df['williams_r'] + k, mode='lines',
                    line=dict(color='blue'), opacity=0.7,
                    name="williams_r")

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()

import numpy as np
import pandas as pd


class DiffPriceLabeller:
    def get_labels(self, data_df: pd.DataFrame,
                   sum_window: int = 3,
                   percents: bool = True) -> tuple[np.ndarray, dict]:
        data = data_df['Close'].to_numpy()
        diff_data = data[1:] - data[:-1]

        diff_data = np.append(diff_data, np.ones(sum_window))
        rolling_window = np.ones(sum_window)
        # sum of the first derivatives
        labels = np.convolve(diff_data, rolling_window, mode='valid')

        if percents:
            labels = labels / data

        debug_info = {}

        return labels, debug_info


if __name__ == "__main__":
    import plotly.graph_objects as go

    candles_df = pd.read_csv("../../price_data/MTLR_1h.csv")

    close_price_data = candles_df['Close'].to_numpy()
    marker_bias = np.mean(np.abs(close_price_data[1:] - close_price_data[:-1]))

    diff_price_labeller = DiffPriceLabeller()
    labels, debug_info = diff_price_labeller.get_labels(data_df=candles_df, sum_window=1, percents=True)

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
    fig.add_candlestick(open=candles_df['Open'], high=candles_df['High'],
                        low=candles_df['Low'], close=candles_df['Close'], opacity=0.5)
    fig.add_scatter(y=close_price_data, mode='lines', line=dict(color='black'), opacity=0.5)
    # labels
    fig.add_scatter(y=labels * 100 + 128, mode='lines', line=dict(color='black'), opacity=0.7)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()

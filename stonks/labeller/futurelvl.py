import numpy as np
import pandas as pd


class FutureLvlLabeller:
    def get_labels(self, data_df: pd.DataFrame,
                   price_type: str = "Close",
                   horizon_len: int = 1,
                   min_differnce: float = 0.0003) -> tuple[np.ndarray, dict]:
        data = data_df[price_type].to_numpy()
        diff_data = data[1:] - data[:-1]

        diff_data = np.append(diff_data, np.ones(horizon_len))
        rolling_window = np.ones(horizon_len)
        # sum of the first derivatives
        labels_raw = np.convolve(diff_data, rolling_window, mode='valid')
        labels_percent = labels_raw / data

        labels = labels_percent.copy()

        labels[labels > min_differnce] = 2
        labels[labels < -min_differnce] = 1
        labels[np.abs(labels) <= min_differnce] = 0

        debug_info = {"labels_percent": labels_percent}

        return labels, debug_info


if __name__ == "__main__":
    import plotly.graph_objects as go

    candles_df = pd.read_csv("../../price_data/MTLR_1h.csv")

    close_price_data = candles_df['Close'].to_numpy()
    marker_bias = np.mean(np.abs(close_price_data[1:] - close_price_data[:-1]))

    future_lvl_labeller = FutureLvlLabeller()
    labels, debug_info = future_lvl_labeller.get_labels(data_df=candles_df,
                                                        price_type="Close",
                                                        horizon_len=3,
                                                        min_differnce=0.0003)

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
                        low=candles_df['Low'], close=candles_df['Close'],
                        name="candles", opacity=0.5)
    fig.add_scatter(y=close_price_data, mode='lines', line=dict(color='black'),
                    opacity=0.5, name="close price")
    # labels
    fig.add_scatter(y=debug_info['labels_percent'] * 100 + 128, mode='lines+markers',
                    line=dict(color='blue'), opacity=0.3,
                    name="future price percent")
    fig.add_scatter(y=labels + 128, mode='lines+markers',
                    line=dict(color='blue'), opacity=0.7,
                    name="future price lvl")
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()

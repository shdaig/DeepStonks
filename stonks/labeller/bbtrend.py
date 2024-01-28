import numpy as np
import zigzag
import pandas as pd


class BBTrendLabeller:
    def get_labels(self, data_df: pd.DataFrame, difference: float = 0.03) -> tuple[np.ndarray, dict]:
        data = data_df['Close'].to_numpy()
        labels_raw = np.array(zigzag.peak_valley_pivots(data, difference, difference * -1))
        labels_raw_args = np.argwhere(labels_raw != 0).flatten()
        labels_raw[labels_raw_args[-1]] = -labels_raw[labels_raw_args[-2]]
        peak_args = np.argwhere(labels_raw == 1).flatten()
        valley_args = np.argwhere(labels_raw == -1).flatten()

        labels = []
        peak_idx = 0
        valley_idx = 0
        label_is_peak = peak_args[peak_idx] < valley_args[valley_idx]
        for i in range(data.shape[0]):
            if label_is_peak:
                if i == peak_args[peak_idx]:
                    labels.append(0)
                    peak_idx += 1
                    label_is_peak = False
                else:
                    labels.append(1)
            else:
                if i == valley_args[valley_idx]:
                    labels.append(1)
                    valley_idx += 1
                    label_is_peak = True
                else:
                    labels.append(0)
        labels = np.array(labels)

        debug_info = {"peak_args": peak_args,
                      "valley_args": valley_args}

        return labels, debug_info


if __name__ == "__main__":
    import plotly.graph_objects as go

    candles_df = pd.read_csv("../../price_data/MTLR_1h.csv")
    close_price_data = candles_df['Close'].to_numpy()

    marker_bias = np.mean(np.abs(close_price_data[1:] - close_price_data[:-1]))

    bbtrend_labeller = BBTrendLabeller()
    labels, debug_info = bbtrend_labeller.get_labels(data_df=candles_df, difference=0.03)

    bearish_labels = np.argwhere(labels == 0).flatten()
    bullish_labels = np.argwhere(labels == 1).flatten()

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
    fig.add_scatter(y=close_price_data, mode='lines', line=dict(color='black'), opacity=0.7)
    # peaks and valleys of zigzag indicator
    fig.add_scatter(x=debug_info["peak_args"], y=close_price_data[debug_info["peak_args"]],
                    mode='markers', marker=dict(color='#4f772d', size=10))
    fig.add_scatter(x=debug_info["valley_args"], y=close_price_data[debug_info["valley_args"]],
                    mode='markers', marker=dict(color='#d00000', size=10))
    # labels
    fig.add_scatter(x=bearish_labels, y=close_price_data[bearish_labels] + marker_bias, mode='markers',
                    marker=dict(color='#d00000', symbol="triangle-down", size=7))
    fig.add_scatter(x=bullish_labels, y=close_price_data[bullish_labels] + marker_bias, mode='markers',
                    marker=dict(color='#4f772d', symbol="triangle-up", size=7))
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()

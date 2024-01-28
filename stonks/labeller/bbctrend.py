import numpy as np
import pandas as pd
from stonks.labeller import bbtrend


class BBCTrendLabeller:
    def get_labels(self, data_df: pd.DataFrame,
                   difference: float = 0.03,
                   integrate_window: int = 5,
                   threshold: float = 0.15) -> tuple[np.ndarray, dict]:
        close_price_data = data_df['Close'].to_numpy()
        bbtrend_labeller = bbtrend.BBTrendLabeller()
        labels, debug_info = bbtrend_labeller.get_labels(data_df=data_df, difference=difference)

        high_price = data_df['High'].to_numpy()
        low_price = data_df['Low'].to_numpy()

        diff_high_price = np.append(np.ediff1d(high_price), 0)
        diff_low_price = np.append(np.ediff1d(low_price), 0)
        diff_high_price = np.abs(diff_high_price)
        diff_low_price = np.abs(diff_low_price)
        # prod_hl_price = diff_high_price * diff_low_price
        diff_stack = np.stack((diff_high_price, diff_low_price), axis=1)
        squared_hl_price = np.max(diff_stack, axis=1) ** 2
        integrated_hl_price = np.convolve(squared_hl_price, np.ones(integrate_window), "same")

        labels[np.argwhere(integrated_hl_price < threshold).flatten()] = 2

        debug_info = {"peak_args": debug_info["peak_args"],
                      "valley_args": debug_info["valley_args"],
                      "close_price_data": close_price_data,
                      "high_price": high_price,
                      "low_price": low_price,
                      "diff_high_price": diff_high_price,
                      "diff_low_price": diff_low_price,
                      "squared_hl_price": squared_hl_price,
                      "integrated_hl_price": integrated_hl_price}

        return labels, debug_info


if __name__ == "__main__":
    import plotly.graph_objects as go

    candles_df = pd.read_csv("../../price_data/MTLR_1h.csv")

    bbctrend_labeller = BBCTrendLabeller()
    labels, debug_info = bbctrend_labeller.get_labels(data_df=candles_df, difference=0.03,
                                                      integrate_window=5, threshold=0.15)

    bearish_labels = np.argwhere(labels == 0).flatten()
    bullish_labels = np.argwhere(labels == 1).flatten()
    consolidation_labels = np.argwhere(labels == 2).flatten()

    close_price_data = debug_info["close_price_data"]

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

    # peaks and valleys of zigzag indicator debug
    fig.add_scatter(x=debug_info["peak_args"], y=close_price_data[debug_info["peak_args"]],
                    mode='markers', marker=dict(color='#4f772d', size=10))
    fig.add_scatter(x=debug_info["valley_args"], y=close_price_data[debug_info["valley_args"]],
                    mode='markers', marker=dict(color='#d00000', size=10))

    # labels
    marker_bias = np.mean(np.abs(close_price_data[1:] - close_price_data[:-1]))
    fig.add_scatter(x=bearish_labels, y=close_price_data[bearish_labels] + marker_bias,
                    mode='markers', marker=dict(color='#d00000', symbol="triangle-down", size=7))
    fig.add_scatter(x=bullish_labels, y=close_price_data[bullish_labels] + marker_bias,
                    mode='markers', marker=dict(color='#4f772d', symbol="triangle-up", size=7))
    fig.add_scatter(x=consolidation_labels, y=close_price_data[consolidation_labels] + marker_bias,
                    mode='markers', marker=dict(color='#5B5B5B', symbol="diamond-wide", size=7))

    # consolidation detection debug
    fig.add_scatter(y=debug_info["high_price"], mode='lines', line=dict(color='black'), opacity=0.3)
    fig.add_scatter(y=debug_info["low_price"], mode='lines', line=dict(color='black'), opacity=0.3)

    fig.add_scatter(y=debug_info["diff_high_price"], mode='lines', line=dict(color='green'))
    fig.add_scatter(y=debug_info["diff_low_price"], mode='lines', line=dict(color='red'))
    fig.add_scatter(y=debug_info["squared_hl_price"] + 1, mode='lines', line=dict(color='blue'))
    fig.add_scatter(y=debug_info["integrated_hl_price"] + 2, mode='lines', line=dict(color='mediumaquamarine'))

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()

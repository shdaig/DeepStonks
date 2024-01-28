import numpy as np
import pandas as pd


class FuturePriceLabeller:
    def get_labels(self, data_df: pd.DataFrame, price_type: str = "Close") -> tuple[np.ndarray, dict]:
        """
        :param data_df: Dataframe with candles
        :param price_type: Close (default) / High / full
        :return: Labels, debug_info
        """
        if price_type == "full":
            features = ["Open", "High", "Low", "Close"]
            data = data_df[features].to_numpy()[1:]
            labels = np.append(data, np.expand_dims(data[-1], axis=0), axis=0)
        else:
            features = price_type
            data = data_df[features].to_numpy()[1:]
            labels = np.append(data, data[-1])

        debug_info = {}

        return labels, debug_info


if __name__ == "__main__":
    import plotly.graph_objects as go

    candles_df = pd.read_csv("../../price_data/MTLR_1h.csv")

    close_price_data = candles_df['Close'].to_numpy()
    marker_bias = np.mean(np.abs(close_price_data[1:] - close_price_data[:-1]))

    diff_price_labeller = FuturePriceLabeller()
    labels_close_only, _ = diff_price_labeller.get_labels(data_df=candles_df)
    labels_full_candle, _ = diff_price_labeller.get_labels(data_df=candles_df, price_type="full")

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
    k = 2
    fig.add_candlestick(open=labels_full_candle[:, 0] + k, high=labels_full_candle[:, 1] + k,
                        low=labels_full_candle[:, 2] + k, close=labels_full_candle[:, 3] + k,
                        opacity=0.2, name="future price in candles")
    fig.add_scatter(y=labels_close_only, mode='lines',
                    line=dict(color='blue'), opacity=0.7,
                    name="future close price")
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()

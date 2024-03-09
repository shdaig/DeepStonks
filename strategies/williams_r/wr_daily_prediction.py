from tinkoff.invest import CandleInterval, Client, InstrumentIdType
from tinkoff.invest.utils import now

import json

from stonks.preprocessing.indicator_generator import IndicatorGenerator
from utils.candles import quotation_to_float
import stonks.utils.token as tkn
import stonks.utils.tickers as tckrs

from datetime import timedelta

import plotly.graph_objects as go

import pandas as pd
import numpy as np


TOKEN = tkn.get_token("../../local_data/token.txt")

if __name__ == "__main__":

    tickers_1d = tckrs.get_tickers_from_file("../../local_data/tickers_kit.txt")
    indicator_generator = IndicatorGenerator()

    with open("best_parameters_12m_all.json", "r") as f:
        best_parameters = json.load(f)

    for ticker in tickers_1d:
        with Client(TOKEN) as client:
            ticker_info = client.instruments.share_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
                                                      class_code="TQBR", id=ticker)
            name = ticker_info.instrument.name
            print(f"name - {ticker_info.instrument.name}")
            print()

        with Client(TOKEN) as client:
            candles = list(client.get_all_candles(
                figi=ticker_info.instrument.figi,
                from_=now() - timedelta(days=300),
                interval=CandleInterval.CANDLE_INTERVAL_DAY
            ))

        dttm_utc = []
        open_price = []
        high_price = []
        low_price = []
        close_price = []
        volume = []

        for candle in candles:
            dttm_utc.append(candle.time)
            open_price.append(quotation_to_float(candle.open))
            high_price.append(quotation_to_float(candle.high))
            low_price.append(quotation_to_float(candle.low))
            close_price.append(quotation_to_float(candle.close))
            volume.append(candle.volume)

        candles_df = pd.DataFrame(
            {
                'dttm_utc': dttm_utc,
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            }
        )

        param = best_parameters[ticker]

        candles_df = indicator_generator.calculate_williams_percent_r(candles_df,
                                                                      smooth_window=param['smooth_williams_r_window'],
                                                                      period=param['williams_r_period'])
        diff_threshold = param["diff_threshold"]
        high_wr_threshold = param["williams_r_thresholds"][0]
        low_wr_threshold = param["williams_r_thresholds"][1]

        buy_points = []
        sell_points = []

        williams_r = candles_df["williams_r"].to_numpy()
        prev_state = "none"
        if williams_r[0] >= high_wr_threshold:
            prev_state = "oversold"
        elif williams_r[0] <= low_wr_threshold:
            prev_state = "overbought"
        # overbought (< 0.2), oversold (> 0.8), none
        for j in range(1, williams_r.shape[0]):
            current_state = "none"
            if williams_r[j] >= high_wr_threshold:
                current_state = "oversold"
            elif williams_r[j] <= low_wr_threshold:
                current_state = "overbought"

            if current_state == "none":
                if prev_state == "oversold":
                    if williams_r[j - 1] - williams_r[j] >= diff_threshold:
                        buy_points.append(j)
                elif prev_state == "overbought":
                    sell_points.append(j)

            prev_state = current_state

        fig = go.Figure()
        fig.add_candlestick(open=candles_df['Open'], high=candles_df['High'],
                            low=candles_df['Low'], close=candles_df['Close'],
                            opacity=0.5, name="candles")
        fig.add_scatter(x=buy_points,
                        y=candles_df['Close'].to_numpy()[buy_points],
                        mode='markers',
                        line=dict(color='green'),
                        opacity=0.7,
                        marker=dict(size=10, symbol="triangle-up"),
                        name="buy")
        fig.add_scatter(x=sell_points,
                        y=candles_df['Close'].to_numpy()[sell_points],
                        mode='markers',
                        line=dict(color='red'),
                        opacity=0.7,
                        marker=dict(size=10, symbol="triangle-down"),
                        name="sell")
        k = candles_df['Close'].mean()
        scale = (np.max(candles_df['Close']) - np.min(candles_df['Close'])) // 5
        if np.max(candles_df['Close']) < 1:
            scale = (np.max(candles_df['Close']) - np.min(candles_df['Close'])) // 5 * 100
        fig.add_scatter(y=np.full(candles_df['williams_r'].shape, high_wr_threshold) * scale + k, mode='lines',
                        line=dict(color='blue'), opacity=0.4,
                        name="williams_r high threshold")
        fig.add_scatter(y=np.full(candles_df['williams_r'].shape, low_wr_threshold) * scale + k, mode='lines',
                        line=dict(color='blue'), opacity=0.4,
                        name="williams_r low threshold")
        fig.add_scatter(y=candles_df['williams_r'] * scale + k, mode='lines',
                        line=dict(color='blue'), opacity=0.7,
                        name="williams_r")
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_layout(title=f"{ticker} - {name}")
        fig.show()

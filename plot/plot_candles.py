import plotly.graph_objects as go

import pandas as pd

X_DATETIME = True

filename = "../price_data/MTLR_1d.csv"
candles_df = pd.read_csv(filename)

if X_DATETIME:
    fig = go.Figure(data=[go.Candlestick(x=candles_df['dttm_utc'],
                    open=candles_df['Open'],
                    high=candles_df['High'],
                    low=candles_df['Low'],
                    close=candles_df['Close'])])
else:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(open=candles_df['Open'],
                    high=candles_df['High'],
                    low=candles_df['Low'],
                    close=candles_df['Close']))

fig.update_layout(title=filename,
                  xaxis_rangeslider_visible=False)
fig.show()

from time import sleep

import pandas as pd

from tinkoff.invest import CandleInterval, Client, InstrumentIdType
from tinkoff.invest.utils import now

from utils.candles import quotation_to_float
from utils.path import mkdir

import stonks.utils.token as tkn
import stonks.utils.tickers as tckrs

from datetime import datetime, timedelta

TOKEN = tkn.get_token("../local_data/token.txt")

DATA_DIR = "../price_data"
mkdir(DATA_DIR)

intervals_tinkoff = {"1m": CandleInterval.CANDLE_INTERVAL_1_MIN,
                     "3m": CandleInterval.CANDLE_INTERVAL_3_MIN,
                     "5m": CandleInterval.CANDLE_INTERVAL_5_MIN,
                     "10m": CandleInterval.CANDLE_INTERVAL_10_MIN,
                     "15m": CandleInterval.CANDLE_INTERVAL_15_MIN,
                     "30m": CandleInterval.CANDLE_INTERVAL_30_MIN,
                     "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
                     "1d": CandleInterval.CANDLE_INTERVAL_DAY}
interval = "1d"
tickers = tckrs.get_tickers_from_file("../local_data/tickers_add.txt")

dt_now = now()

for ticker in tickers:
    with Client(TOKEN) as client:
        ticker_info = client.instruments.share_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
                                                  class_code="TQBR", id=ticker)
        print(f"name - {ticker_info.instrument.name}")
        print(f"figi - {ticker_info.instrument.figi}")
        print(f"first_1min_candle_date - {ticker_info.instrument.first_1min_candle_date}")
        print()
        print(ticker_info)

    with Client(TOKEN) as client:
        candles = list(client.get_all_candles(
            figi=ticker_info.instrument.figi,
            from_=dt_now - timedelta(days=365*5),
            interval=intervals_tinkoff[interval],
        ))

    dttm_utc = []
    open_price = []
    high_price = []
    low_price = []
    close_price = []
    volume = []
    hour = []
    dayofweek = []

    now = datetime.now()

    if now.hour >= 21 or now.hour <= 7:
        for candle in candles:
            dttm_utc.append(candle.time)
            open_price.append(quotation_to_float(candle.open))
            high_price.append(quotation_to_float(candle.high))
            low_price.append(quotation_to_float(candle.low))
            close_price.append(quotation_to_float(candle.close))
            volume.append(candle.volume)
            hour.append(candle.time.hour)
            dayofweek.append(candle.time.weekday())
    else:
        for candle in candles[:-1]:
            dttm_utc.append(candle.time)
            open_price.append(quotation_to_float(candle.open))
            high_price.append(quotation_to_float(candle.high))
            low_price.append(quotation_to_float(candle.low))
            close_price.append(quotation_to_float(candle.close))
            volume.append(candle.volume)
            hour.append(candle.time.hour)
            dayofweek.append(candle.time.weekday())

    stocks_df = pd.DataFrame(
        {
            'dttm_utc': dttm_utc,
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume,
            'hour': hour,
            'dayofweek': dayofweek
        }
    )

    stocks_df.to_csv(f"{DATA_DIR}/{ticker}_{interval}.csv", index=False)
    print('done!\n')

    sleep(10)






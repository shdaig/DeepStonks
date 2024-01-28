import json

import pandas as pd
import numpy as np

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
    tickers_1d = ["TRNFP", "RKKE", "VSMO",
                  "UNKL", "LNZL", "CHMK",
                  "KRKNP", "GMKN", "PLZL",
                  "AKRN", "KROT", "HHRU",
                  "LNZLP", "LKOH", "MGNT",
                  "BELU", "PHOR", "TCSG",
                  "GCHE", "NKHP", "SMLT",
                  "OZON", "FIVE", "YNDX",
                  "SBER", "TATN", "MTLR",
                  "MRKZ", "VKCO", "GLTR"]

    best_parameters = dict()

    for ticker in tickers_1d:
        candles_df = pd.read_csv(f"../../data/{ticker}_1d.csv")

        params = {"smooth_williams_r_window": [1, 2, 3, 4, 5, 6, 7],
                  "williams_r_period": [8, 10, 12, 14, 16, 18, 20, 22],
                  "diff_threshold": [0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
                                     0.06, 0.07, 0.08, 0.09, 0.10,
                                     0.11, 0.12, 0.13, 0.14, 0.15],
                  "williams_r_thresholds": [[0.8, 0.2], [0.775, 0.225], [0.75, 0.25], [0.725, 0.275], [0.7, 0.3]]}
        indicator_generator = IndicatorGenerator()

        candles_dfs = {}
        for window in params["smooth_williams_r_window"]:
            for period in params["williams_r_period"]:
                candles_dfs[f"{window}_{period}"] = indicator_generator.calculate_williams_percent_r(candles_df,
                                                                                                    smooth_window=window,
                                                                                                    period=period)

        param_grid = ParameterGrid(params)

        results_6m = []
        results_1y = []
        results_1y6m = []
        results_2y = []
        results_3y = []

        for param in param_grid:
            candles_df = candles_dfs[f"{param['smooth_williams_r_window']}_{param['williams_r_period']}"]
            diff_threshold = param["diff_threshold"]

            data_size = candles_df.shape[0]

            test_m_period = 3

            separate_parts = data_size // (30 * test_m_period) - 1

            chunk_len = 30 * test_m_period
            pad = 20

            buy_stock = 1
            commission = 0.0005

            total_percents = []

            high_wr_threshold = param["williams_r_thresholds"][0]
            low_wr_threshold = param["williams_r_thresholds"][1]

            for i in range(separate_parts):
                cash = 1000000.0
                stocks_count = 0
                last_cash = cash

                start_idx = data_size - chunk_len * (i + 1) - pad
                end_idx = data_size - chunk_len * i
                # print(f"[{i}] {start_idx} - {end_idx}: (len {end_idx - start_idx})")

                chunk = candles_df.iloc[start_idx:end_idx].copy()
                buy_points = []
                sell_points = []
                bought = False

                williams_r = chunk["williams_r"].to_numpy()
                prev_state = "none"
                if williams_r[pad - 1] >= high_wr_threshold:
                    prev_state = "oversold"
                elif williams_r[pad - 1] <= low_wr_threshold:
                    prev_state = "overbought"
                # overbought (< 0.2), oversold (> 0.8), none
                spends_total = 0
                spends_temp = 0
                profit_total = 0
                for j in range(pad, pad + chunk_len - 1):
                    current_state = "none"
                    if williams_r[j] >= high_wr_threshold:
                        current_state = "oversold"
                    elif williams_r[j] <= low_wr_threshold:
                        current_state = "overbought"

                    if current_state == "none":
                        if prev_state == "oversold":
                            if williams_r[j - 1] - williams_r[j] >= diff_threshold:
                                current_price = (chunk["Open"].iloc[j + 1] + chunk["Close"].iloc[j + 1]) / 2
                                cash, spends = buy(cash, current_price, buy_stock, commission=commission)
                                stocks_count += buy_stock
                                spends_temp += spends
                                buy_points.append(j)
                                bought = True
                                k = 0
                        elif prev_state == "overbought":
                            if bought:
                                current_price = (chunk["Open"].iloc[j + 1] + chunk["Close"].iloc[j + 1]) / 2
                                cash, profit = sell(cash, current_price, stocks_count, commission=commission)
                                stocks_count = 0
                                last_cash = cash
                                profit_total += profit
                                spends_total += spends_temp
                                spends_temp = 0
                                sell_points.append(j)
                                bought = False
                                k = 0
                    prev_state = current_state

                total_percent = 0.0
                if spends_total > 0:
                    total_percent = profit_total / spends_total - 1.0
                total_percents.append(total_percent)

            time_scale = 6 // test_m_period

            results_6m.append(np.mean(total_percents[: time_scale]) / test_m_period)
            results_1y.append(np.mean(total_percents[: 2 * time_scale]) / test_m_period)
            results_1y6m.append(np.mean(total_percents[: 3 * time_scale]) / test_m_period)

            if separate_parts >= 4 * time_scale:
                results_2y.append(np.mean(total_percents[2 * time_scale: 4 * time_scale]) / test_m_period)

            if separate_parts >= 6 * time_scale:
                results_3y.append(np.mean(total_percents[4 * time_scale: 6 * time_scale]) / test_m_period)

        print(f"\nSummary {ticker}:")

        # sort_arg = np.argmax(results_6m)
        sort_arg = np.argmax(results_1y)
        # sort_arg = np.argmax(results_1y6m)

        results_6m_round = round(results_6m[sort_arg] * 100, 2)
        print(f"\tAvg %/month for 6 months: {results_6m_round} %")
        results_1y_round = round(results_1y[sort_arg] * 100, 2)
        print(f"\tAvg %/month for 1 years: {results_1y_round} %")
        results_1y6m_round = round(results_1y6m[sort_arg] * 100, 2)
        print(f"\tAvg %/month for 1.5 years: {results_1y6m_round} %")
        if len(results_2y) > 0:
            results_2y_round = round(results_2y[sort_arg] * 100, 2)
            print(f"\tAvg %/month for 2nd year: {results_2y_round} %")
        if len(results_3y) > 0:
            results_3y_round = round(results_3y[sort_arg] * 100, 2)
            print(f"\tAvg %/month for 3rd year: {results_3y_round} %")

        print(f"Best parameters {ticker}:")
        print(param_grid[sort_arg])

        best_parameters[ticker] = param_grid[sort_arg]

    with open('best_parameters_12m.json', 'w') as f:
        json.dump(best_parameters, f)

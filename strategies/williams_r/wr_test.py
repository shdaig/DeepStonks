import json

import pandas as pd
import numpy as np

from stonks.preprocessing.indicator_generator import IndicatorGenerator
from sklearn.model_selection import ParameterGrid

import stonks.utils.tickers as tckrs


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


def find_firs_idx(data, period):
    idx = np.argwhere(np.array(data) >= period)
    if len(idx) == 0:
        return -1
    return idx[0][0]


def process_period_trades(period_trades, period=0):
    result = np.sum(period_trades)
    if period != 0:
        result /= period
    return result


if __name__ == "__main__":
    tickers_1d = tckrs.get_tickers_from_file("../../local_data/tickers_all.txt")

    best_parameters = dict()

    for ticker in tickers_1d:
        candles_df = pd.read_csv(f"../../price_data/{ticker}_1d.csv")

        params = {"smooth_williams_r_window": [1, 2, 3, 4, 5, 6, 7],
                  "williams_r_period": [8, 10, 12, 14, 16, 18, 20, 22],
                  "diff_threshold": [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
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

        times = []
        trade_results = []

        data_size = candles_df.shape[0]
        start_idx = 30
        end_idx = data_size

        for param in param_grid:
            candles_df = candles_dfs[f"{param['smooth_williams_r_window']}_{param['williams_r_period']}"]
            diff_threshold = param["diff_threshold"]
            high_wr_threshold = param["williams_r_thresholds"][0]
            low_wr_threshold = param["williams_r_thresholds"][1]

            buy_stock = 1
            commission = 0.0005
            cash = 1000000.0
            stocks_count = 0

            buy_points = []
            sell_points = []
            bought = False

            param_times = []
            param_trade_results = []

            williams_r = candles_df["williams_r"].to_numpy()
            prev_state = "none"
            if williams_r[start_idx - 1] >= high_wr_threshold:
                prev_state = "oversold"
            elif williams_r[start_idx - 1] <= low_wr_threshold:
                prev_state = "overbought"
            # overbought (< 0.2), oversold (> 0.8), none
            spends_total = 0
            for j in range(start_idx, end_idx - 1):
                current_state = "none"
                if williams_r[j] >= high_wr_threshold:
                    current_state = "oversold"
                elif williams_r[j] <= low_wr_threshold:
                    current_state = "overbought"

                if current_state == "none":
                    if prev_state == "oversold":
                        if williams_r[j - 1] - williams_r[j] >= diff_threshold:
                            current_price = (candles_df["Open"].iloc[j + 1] + candles_df["Close"].iloc[j + 1]) / 2
                            cash, spends = buy(cash, current_price, buy_stock, commission=commission)
                            stocks_count += buy_stock
                            spends_total += spends
                            buy_points.append(j)
                            bought = True
                    elif prev_state == "overbought":
                        if bought:
                            current_price = (candles_df["Open"].iloc[j + 1] + candles_df["Close"].iloc[j + 1]) / 2
                            cash, profit = sell(cash, current_price, stocks_count, commission=commission)
                            stocks_count = 0
                            sell_points.append(j)
                            total_percent = 0.0
                            if spends_total > 0:
                                total_percent = profit / spends_total - 1.0
                            param_times.append(j)
                            param_trade_results.append(total_percent)
                            spends_total = 0
                            bought = False
                prev_state = current_state

            times.append(param_times)
            trade_results.append(param_trade_results)

        results_6m = []
        trade_count_6m = []
        positive_trade_count_6m = []

        results_12m = []
        trade_count_12m = []
        positive_trade_count_12m = []

        results_6m12m = []
        trade_count_6m12m = []
        positive_trade_count_6m12m = []

        results_6m18m = []
        trade_count_6m18m = []
        positive_trade_count_6m18m = []

        for i in range(len(times)):
            period_1m = 30
            idx_6m = find_firs_idx(times[i], data_size - period_1m * 6)
            idx_12m = find_firs_idx(times[i], data_size - period_1m * 12)
            idx_18m = find_firs_idx(times[i], data_size - period_1m * 18)

            if idx_6m != -1:
                period_trade_results = trade_results[i][idx_6m:]
                if len(period_trade_results) > 0:
                    mean_profit = process_period_trades(period_trade_results, 6)
                    results_6m.append(mean_profit)
                    positive_trade_count_6m.append(np.sum(np.array(period_trade_results) > 0.0))
                else:
                    results_6m.append(0.0)
                    positive_trade_count_6m.append(0)
                trade_count_6m.append(len(period_trade_results))
            else:
                results_6m.append(0.0)
                trade_count_6m.append(0)
                positive_trade_count_6m.append(0)

            if idx_12m != -1:
                period_trade_results = trade_results[i][idx_12m:]
                if len(period_trade_results) > 0:
                    mean_profit = process_period_trades(period_trade_results, 12)
                    results_12m.append(mean_profit)
                    positive_trade_count_12m.append(np.sum(np.array(period_trade_results) > 0.0))
                else:
                    results_12m.append(0.0)
                    positive_trade_count_12m.append(0)
                trade_count_12m.append(len(period_trade_results))
            else:
                results_12m.append(0.0)
                trade_count_12m.append(0)
                positive_trade_count_12m.append(0)

            if idx_12m != -1:
                period_trade_results = np.array(trade_results[i])[idx_12m:idx_6m]
                if idx_6m == -1:
                    period_trade_results = np.array(trade_results[i])[idx_12m:]
                if len(period_trade_results) > 0:
                    mean_profit = process_period_trades(period_trade_results, 6)
                    results_6m12m.append(mean_profit)
                    positive_trade_count_6m12m.append(np.sum(np.array(period_trade_results) > 0.0))
                else:
                    results_6m12m.append(0.0)
                    positive_trade_count_6m12m.append(0)
                trade_count_6m12m.append(len(period_trade_results))
            else:
                results_6m12m.append(0.0)
                trade_count_6m12m.append(0)
                positive_trade_count_6m12m.append(0)

            if idx_18m != -1:
                period_trade_results = np.array(trade_results[i])[idx_18m:idx_6m]
                if idx_6m == -1:
                    period_trade_results = np.array(trade_results[i])[idx_18m:]
                if len(period_trade_results) > 0:
                    mean_profit = process_period_trades(period_trade_results, 12)
                    results_6m18m.append(mean_profit)
                    positive_trade_count_6m18m.append(np.sum(np.array(period_trade_results) > 0.0))
                else:
                    results_6m18m.append(0.0)
                    positive_trade_count_6m18m.append(0)
                trade_count_6m18m.append(len(period_trade_results))
            else:
                results_6m18m.append(0.0)
                trade_count_6m18m.append(0)
                positive_trade_count_6m18m.append(0)

        print(f"\nSummary {ticker}:")

        # sort_arg = np.argmax(results_6m)
        sort_arg = np.argmax(results_12m)
        # sort_arg = np.argmax(results_6m12m)
        # sort_arg = np.argmax(results_6m18m)
        # sort_arg = np.argmax(positive_trade_count_6m12m)
        # sort_arg = np.argmax(positive_trade_count_12m)

        results_6m_round = round(results_6m[sort_arg] * 100, 2)
        print(f"\tAvg %/month for 6 months: {results_6m_round} % (count: {trade_count_6m[sort_arg]}, +: {positive_trade_count_6m[sort_arg]})")
        results_6m12m_round = round(results_6m12m[sort_arg] * 100, 2)
        print(f"\tAvg %/month from 12 to 6 months: {results_6m12m_round} % (count: {trade_count_6m12m[sort_arg]}, +: {positive_trade_count_6m12m[sort_arg]})")
        results_12m_round = round(results_12m[sort_arg] * 100, 2)
        print(f"\tAvg %/month for 1 year: {results_12m_round} % (count: {trade_count_12m[sort_arg]}, +: {positive_trade_count_12m[sort_arg]})")
        results_6m18m_round = round(results_6m18m[sort_arg] * 100, 2)
        print(f"\tAvg %/month from 18 to 6 months {results_6m18m_round} % (count: {trade_count_6m18m[sort_arg]}, +: {positive_trade_count_6m18m[sort_arg]})")

        print(f"Best parameters {ticker}:")
        print(param_grid[sort_arg])

        best_parameters[ticker] = param_grid[sort_arg]

    with open('best_parameters_12m_all.json', 'w') as f:
        json.dump(best_parameters, f)

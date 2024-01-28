import pandas as pd
import numpy as np


def __normalize(data):
    return data / np.sqrt(np.sum(np.power(data, 2)))


# Нормализованная цена закрытия + категориальные день и час
def convert_close_price_dayhour(df, s):
    x = []
    y = []

    hours = pd.get_dummies(df["hour"]).astype(int)
    dayofweeks = pd.get_dummies(df["dayofweek"]).astype(int)

    for i in range(0, df.shape[0] - s):
        x_norm = __normalize(df.loc[i : i + s - 1, ["Close"]].values.flatten())
        x_norm = np.concatenate((x_norm, dayofweeks.iloc[i + s - 1].values))
        x_norm = np.concatenate((x_norm, hours.iloc[i + s - 1].values))
        x.append(x_norm)
        y.append(df.iloc[i + s - 1].label)

    x = np.array(x)
    y = np.array(y)
    
    return x, y


# Только нормализованная цена закрытия
def convert_close_price(df, s):
    x = []
    y = []

    for i in range(0, df.shape[0] - s):
        x_norm = __normalize(df.loc[i : i + s - 1, ["Close"]].values.flatten())
        x.append(x_norm)
        y.append(df.iloc[i + s - 1].label)

    x = np.array(x)
    y = np.array(y)
    
    return x, y


# Вытягивает вектор из всех параметров
def convert_all_parameters(df, s):
    x = []
    y = []

    for i in range(0, df.shape[0] - s):
        open_norm = __normalize(df.loc[i : i + s - 1, ["Open"]].values.flatten())
        high_norm = __normalize(df.loc[i : i + s - 1, ["High"]].values.flatten())
        low_norm = __normalize(df.loc[i : i + s - 1, ["Low"]].values.flatten())
        close_norm = __normalize(df.loc[i : i + s - 1, ["Close"]].values.flatten())
        
        volume = df.loc[i : i + s - 1, ["Volume"]].values.flatten()
        hour = df.loc[i : i + s - 1, ["hour"]].values.flatten()
        dayofweek = df.loc[i : i + s - 1, ["dayofweek"]].values.flatten()
        
        df_norm = pd.DataFrame(
            {
                'Open': open_norm, 
                'High': high_norm, 
                'Low': low_norm,
                'Close': close_norm,
                'Volume': volume / 100000,
                'hour': hour,
                'dayofweek': dayofweek
            }
        )
        
        x.append(df_norm.loc[:, ["Open", "High", "Low", "Close", "Volume", "hour", "dayofweek"]].values.flatten())
        y.append(df.iloc[i + s - 1].label)

    x = np.array(x)
    y = np.array(y)
    
    return x, y


# Вытягивает вектор из ценовых параметров
def convert_all_price(df, s):
    x = []
    y = []

    for i in range(0, df.shape[0] - s):
        open_norm = __normalize(df.loc[i : i + s - 1, ["Open"]].values.flatten())
        high_norm = __normalize(df.loc[i : i + s - 1, ["High"]].values.flatten())
        low_norm = __normalize(df.loc[i : i + s - 1, ["Low"]].values.flatten())
        close_norm = __normalize(df.loc[i : i + s - 1, ["Close"]].values.flatten())
        
        df_norm = pd.DataFrame(
            {
                'Open': open_norm, 
                'High': high_norm, 
                'Low': low_norm,
                'Close': close_norm
            }
        )
        
        x.append(df_norm.loc[:, ["Open", "High", "Low", "Close"]].values.flatten())
        y.append(df.iloc[i + s - 1].label)

    x = np.array(x)
    y = np.array(y)
    
    return x, y


# Вытягивает вектор из всех параметров + разбивает категориальные признаки на категории
def convert_all_categorical(df, s):
    x = []
    y = []

    df_copy = df.copy()

    hours = pd.get_dummies(df_copy["hour"]).astype(int)
    dayofweeks = pd.get_dummies(df_copy["dayofweek"]).astype(int)

    df_copy = df_copy.drop(columns=["hour", "dayofweek"])

    df_copy = pd.concat([df_copy, hours, dayofweeks], axis=1)

    for i in range(0, df.shape[0] - s):
        open_norm = __normalize(df_copy.loc[i : i + s - 1, ["Open"]].values.flatten())
        high_norm = __normalize(df_copy.loc[i : i + s - 1, ["High"]].values.flatten())
        low_norm = __normalize(df_copy.loc[i : i + s - 1, ["Low"]].values.flatten())
        close_norm = __normalize(df_copy.loc[i : i + s - 1, ["Close"]].values.flatten())

        volume = df_copy.loc[i : i + s - 1, ["Volume"]].values.flatten()

        df_norm = pd.DataFrame(
            {
                'Open': open_norm, 
                'High': high_norm, 
                'Low': low_norm,
                'Close': close_norm,
                'Volume': volume / 1000000
            }
        )

        df_hw = df_copy.iloc[i : i + s, 7:]
        df_hw = df_hw.reset_index(drop=True)

        df_norm = pd.concat([df_norm, df_hw], axis=1)

        x.append(df_norm.values.flatten())
        y.append(df.iloc[i + s - 1].label)

    x = np.array(x)
    y = np.array(y)
    
    return x, y


# Генерирует матрицу из параметров
def convert_multidim(df, s):
    x = []
    y = []

    for i in range(0, df.shape[0] - s):
        x.append(df.loc[i : i + s - 1, ["Open", "High", "Low", "Close", "Volume", "hour", "dayofweek"]].values)
        y.append(df.iloc[i + s - 1].label)

    x = np.array(x)
    y = np.array(y)
    
    return x, y
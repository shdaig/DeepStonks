import numpy as np
import pandas as pd

from stonks.labeller import bbctrend


class SampleFormer:
    def _minmax_normalization(self, data: np.ndarray) -> np.ndarray:
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def _l2_normalization(self, data: np.ndarray) -> np.ndarray:
        return data / np.sqrt(np.sum(np.power(data, 2)))

    def get_samples(self, candles_df: pd.DataFrame,
                    depth: int,
                    features_raw: list,
                    normalization_features: list = None,
                    scaling_features: list = None,
                    vectorized: bool = False,
                    normalization: str = None) -> tuple[np.ndarray, np.ndarray]:
        features = features_raw.copy() + ["labels"]

        norm_features_idx = []
        # scaling_features_idx = []
        if normalization_features is not None:
            for feature in normalization_features:
                norm_features_idx.append(features.index(feature))
        # if scaling_features is not None:
        #     for feature in scaling_features:
        #         scaling_features_idx.append(features.index(feature))

        features_slice = candles_df[features].copy()

        if scaling_features is not None:
            for feature in scaling_features:
                feature_data = features_slice[feature]

                feature_mean = feature_data.mean()
                feature_std = feature_data.std()

                features_slice[feature] = (feature_data - feature_mean) / feature_std

        rolling_raw = features_slice.rolling(depth)
        windows = [window.to_numpy() for window in rolling_raw]
        windows = np.array(windows[depth:])

        y = windows[:, -1, -1]
        x = windows[:, :, :-1]

        if normalization == "minmax":
            x[:, :, norm_features_idx] = np.array([self._minmax_normalization(subarray) for subarray in x[:, :, norm_features_idx]])
        elif normalization == "l2":
            x[:, :, norm_features_idx] = np.array([self._l2_normalization(subarray) for subarray in x[:, :, norm_features_idx]])

        if vectorized:
            if len(features) > 1:
                x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
            else:
                x = x.reshape(x.shape[0], -1)

        return x, y


if __name__ == "__main__":
    sampleformer = SampleFormer()
    bbctrend_labeller = bbctrend.BBCTrendLabeller()

    candles_df = pd.read_csv("../../price_data/MTLR_1h.csv")
    labels, debug_info = bbctrend_labeller.get_labels(data_df=candles_df, difference=0.03,
                                                      integrate_window=5, threshold=0.15)
    candles_df['labels'] = pd.Series(labels)

    x, y = sampleformer.get_samples(candles_df, 5,
                                    ["Close", "Open", "High", "Low", "Volume"],
                                    vectorized=False, normalization=None)
    print(x[0])
    print(x.shape)
    print(y.shape)

    x, y = sampleformer.get_samples(candles_df, 5,
                                    ["Close", "Open", "High", "Low", "Volume"],
                                    ["Close", "Open", "High", "Low"],
                                    ["Volume"],
                                     normalization="minmax")
    print(x[0])
    print(x.shape)
    print(y.shape)

    x, y = sampleformer.get_samples(candles_df, 5,
                                    ["Close", "Open", "High", "Low", "Volume"],
                                    ["Close", "Open", "High", "Low"],
                                    normalization="minmax")
    print(x[0])
    print(x.shape)
    print(y.shape)

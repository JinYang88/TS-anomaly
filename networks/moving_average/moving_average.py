import numpy as np
from IPython import embed


class MovingAverage:
    def __init__(self, anomaly_ts_num: float = 0.5):
        self.anomaly_ts_num = anomaly_ts_num
        self.diff_upper_threshold_list = []

    def _moving_average(self, ts, w):
        assert len(ts.shape) == 1, 'Only support 1-d time series currently'
        ma_result = np.zeros(ts.shape)
        ma_result[w - 1:] = np.convolve(ts, np.ones(w), 'valid') / w
        ma_result[:w - 1] = ts[:w - 1]
        return ma_result

    def fit(self, train_data, w: int = 5):
        """
        params
        w: windows size
        train_data: type is np.ndarray
        """
        self.w = w
        for i in range(train_data.shape[1]):
            ma_result = self._moving_average(train_data[:, i], w)
            ma_diff = np.abs(ma_result - train_data[:, i])
            mean = np.mean(ma_diff)
            std = np.std(ma_diff)
            self.diff_upper_threshold_list.append(mean + 3 * std)

    def predict(self, test_data, w: int = 5):
        """
        test_data: type is np.ndarray
        """
        anomaly_indice = np.zeros(test_data.shape[0])
        for dim in range(test_data.shape[1]):
            ts = test_data[:, dim]
            ma_result = self._moving_average(ts, w)
            ma_diff = np.abs(ma_result - ts)
            upper = self.diff_upper_threshold_list[dim]
            anomaly_dim_indice = (ma_diff > upper).astype(int)
            anomaly_indice += anomaly_dim_indice
        anomaly = (anomaly_indice > int(
            self.anomaly_ts_num * test_data.shape[1])).astype(bool)
        return anomaly

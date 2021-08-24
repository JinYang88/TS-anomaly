import numpy as np
from IPython import embed
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    def __init__(self, anomaly_ts_num: float = 0.5):
        self.anomaly_ts_num = anomaly_ts_num
        self.diff_upper_threshold_list = []

    def _arima_model(self, ts):
        assert len(ts.shape) == 1, 'Only support 1-d time series currently'
        ar_result = np.zeros(ts.shape)
        model = ARIMA(ts, order=(1, 0, 0))
        model_fit = model.fit()
        fittedLen = model_fit.fittedvalues.shape[0]
        ar_result[ts.shape[0] - fittedLen:] = model_fit.fittedvalues
        ar_result[:ts.shape[0] - fittedLen] = ts[:ts.shape[0] - fittedLen]
        return ar_result

    def fit(self, train_data):
        """
        train_data: type is np.ndarray
        """
        for i in range(train_data.shape[1]):
            print(f"fitting dim {i}")
            ar_result = self._arima_model(train_data[:, i])
            ar_diff = np.abs(ar_result - train_data[:, i])
            mean = np.mean(ar_diff)
            std = np.std(ar_diff)
            self.diff_upper_threshold_list.append(mean + 3 * std)

    def predict(self, test_data, w: int = 5):
        """
        test_data: type is np.ndarray
        """
        anomaly_indice = np.zeros(test_data.shape[0])
        for dim in range(test_data.shape[1]):
            print(f"predicting dim {dim}")
            ts = test_data[:, dim]
            ar_result = self._arima_model(ts)
            ar_diff = np.abs(ar_result - ts)
            upper = self.diff_upper_threshold_list[dim]
            anomaly_dim_indice = (ar_diff > upper).astype(int)
            anomaly_indice += anomaly_dim_indice
        anomaly = (anomaly_indice > int(
            self.anomaly_ts_num * test_data.shape[1])).astype(bool)
        return anomaly

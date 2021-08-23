import numpy as np
from pyod.models.iforest import IForest
from IPython import embed


class IForestUni:
    def __init__(self, anomaly_ts_num: float = 0.5, anomaly_threshold=0.2):
        self.anomaly_ts_num = anomaly_ts_num
        self.anomaly_threshold = anomaly_threshold
        self.iForestModels = []

    def fit(self, train_data, n_estimators=10):
        """
        train_data: type is np.ndarray
        """
        for i in range(train_data.shape[1]):
            ts = train_data[:, i].copy()
            od = IForest(n_estimators=n_estimators)
            od.fit(ts.reshape(-1, 1))
            self.iForestModels.append(od)

    def predict(self, test_data):
        """
        test_data: type is np.ndarray
        """
        anomaly_indice = np.zeros(test_data.shape[0])
        for dim in range(test_data.shape[1]):
            ts = test_data[:, dim].copy()
            anomaly_score = self.iForestModels[dim].decision_function(
                ts.reshape(-1, 1))
            anomaly_dim_indice = (
                anomaly_score > self.anomaly_threshold).astype(int)
            anomaly_indice += anomaly_dim_indice
        anomaly = (anomaly_indice > int(
            self.anomaly_ts_num * test_data.shape[1])).astype(bool)
        return anomaly

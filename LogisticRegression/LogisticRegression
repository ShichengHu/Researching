import numpy as np
from Util.Bases import RegressionBase


class LogisticRegression(RegressionBase):
    """
    用梯度下降法实现逻辑回归（实现二分类）

    类别空间是0，1
    """
    def __init__(self, **kwargs):
        RegressionBase.__init__(self)
        self._x = self._y = None
        self._theta = None
        self._w = kwargs.get("w", None)
        self._b = kwargs.get("b", None)

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def _feed_data(self, x, y):
        """
        数据预处理

        :params _x : 对x补一列全一作为新的x,这样就把w,b一起处理了
        :param x: array(n_sample, n_features)
        :param y: array(n_samples, 1)
        :return: 对数据的预处理
        """
        self._x = np.hstack((x, np.ones(len(y)).reshape(-1, 1)))  # 合并
        self._y = np.array(y)
        m, n = self._x.shape
        self._w = np.zeros(n-1)
        self._b = 0.
        self._theta = np.zeros(n)

    def fit(self, x, y, sample_weight=None, **kwargs):
        """
        训练

        :params epoch ,lr : 可以由外界输入
        :param x: 同上
        :param y:
        :param sample_weight: 权重
        :param kwargs:
        :return: 获得w, b
        """
        epoch = kwargs.get("epoch", 10**4)  # 循环次数
        lr = kwargs.get("lr", 1e-18)  # 步距
        x, y = np.atleast_2d(x), np.atleast_1d(y)
        # 初始化参数
        self._feed_data(x, y)
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.array(sample_weight)

        for _ in range(epoch):
            print(_)
            tmp = self._theta  # 用批量梯度下降法寻找θ
            self._theta += np.average(
                sample_weight * (self._y - np.sum(self.sigmoid(self._x * self._theta), axis=1)) * self._x.T, axis=-1)
            # 提高算法的稳健性，加入这个判决
            if np.linalg.norm(self._theta - tmp) <= lr:
                break
            # 更新
        self._w = self._theta[: -1]
        self._b = self._theta[-1]

    def predict(self, x, get_raw_result=False):
        y_pred = np.sum(self.sigmoid(self._w * x + self._b), axis=1)
        return y_pred if get_raw_result else np.round(y_pred)  # np.sign(y_pred)

import numpy as np
from copy import deepcopy
from Util.Bases import KernelBase, KernelConfig
# 更新，在__init__()中又加了**kwargs,为了在集成学习的时候用


class SVM(KernelBase):
    """
    引入软间隔，正则化，正则化项是hinge损失。引入核函数，用SMO算法更新松弛变量
    """
    def __init__(self, **kwargs):
        KernelBase.__init__(self)
        # 对于SVM而言，循环体中允许存在误差
        self._fit_args, self._fit_args_names = [1e-3], ["tol"]
        # 初始化给定这两个参数完全没法代入，因为有for name in kwargs rather than fit_args_names
        self._c = kwargs.get("_c", None)

    @staticmethod
    def optimizer_object(k11, k12, k22, y1, y2, a1, a2, e1, e2, b):
        """
        计算最小化对象的临界值
        :param k11:
        :param k12:
        :param k22:
        :param y1:
        :param y2:
        :param a1: 挑选的两个α
        :param a2:
        :param e1:对于的预测误差
        :param e2:
        :param b:
        :return:欲最小化的对象
        """
        s = y1*y2
        f1 = y1 * (e1 - b) - a1 * k11 - a2 * s * k12
        f2 = y1 * (e2 - b) - s * a1 * k12 - a2 * k22
        op = (k11 * a1 ** 2 + 2 * k12 * s * a1 * a2 + k22 * a2 ** 2) * 0.5 + a1 * f1 + a2 * f2
        return op

    # SMO 挑选第一个样本
    def _pick_first(self, tol):
        con1 = self._alpha > 0  # 全是False
        con2 = self._alpha < self._c  # 全是True
        # 损失向量
        assert len(self._y) == len(self._prediction_cache), "ERROR"
        err1 = self._y * self._prediction_cache - 1  # 预测正确，为0，否则为-2  全是 -1
        err2 = deepcopy(err1)
        err3 = deepcopy(err1)

        # err1[(con1 | err1 >= 0)] = 0  # alpha>0 或 err1>=0, prediction_cache=0,直接炸
        # err2[(~con1 | ~con2) | (err2 == 0)] = 0
        # err3[(con2 | err3 <= 0)] = 0

        err1[(con1 & (err1 <= 0)) | (~con1 & (err1 > 0))] = 0
        err2[((~con1 | ~con2) & (err2 != 0)) | ((con1 & con2) & (err2 == 0))] = 0
        err3[(con2 & (err3 >= 0)) | (~con2 & (err3 < 0))] = 0

        err = err1 ** 2 + err2 ** 2 + err3 ** 2  # np.sqrt()/1e5
        idx = np.argmax(err)  # 误差最大的都比tol小，训练结束，返回真值为None
        if err[idx] < tol:  # 每次都是直接结束，让人无奈
            return
        return idx

    # 挑选第二个样本,随机取
    def _pick_second(self, idx1):
        idx = np.random.randint(len(self._y))
        while idx == idx1:
            idx = self._pick_second(idx1)
        return idx

    def get_lower_bound(self, idx1, idx2):
        # alpha2
        if self._y[idx1] == self._y[idx2]:
            return max(0, self._alpha[idx2] + self._alpha[idx1] - self._c)  # 0 -> 1e-10
        return max(0, self._alpha[idx2] - self._alpha[idx1])  # 0 -> 1e-10

    def get_upper_bound(self, idx1, idx2):
        if self._y[idx1] == self._y[idx2]:
            return min(self._c, self._alpha[idx2] + self._alpha[idx1])
        return min(self._c, self._c + self._alpha[idx2] - self._alpha[idx1])

    def _update_dw_cache(self, idx1, idx2, da1, da2, y1, y2):
        self._dw_cache = np.array([da1*y1, da2*y2])
        self._w[idx1] += self._dw_cache[0]  # 0
        self._w[idx2] += self._dw_cache[1]  # 1

    def _update_db_cache(self, idx1, idx2, da1, da2, y1, y2, e1, e2):
        # 如果这两个样本均满足0<alpha<C, 则b1 = b2,否则 b = (b1 + b2)/2
        gram_11, gram_12, gram_22 = self._gram[idx1, idx1], self._gram[idx1, idx2], self._gram[idx2, idx2]
        b1 = -e1 - y1 * gram_11 * da1 - y2*gram_12 * da2
        b2 = -e2 - y1*gram_12 * da1 - y2*gram_22 * da2
        # 尽量取界内，不满足取界上
        if self._c > self._alpha[idx2] > 0:  # alpha1，alpha2 或alpha2满足取值在界内b=b1=b2
            self._db_cache = b2
        elif self._c > self._alpha[idx1] > 0:  # alpha1不满足取在界内，alpha2不在界内
            self._db_cache = b1
        else:  # 都不在界内，可能在界上
            self._db_cache = (b1 + b2) / 2
        # 更新self._b
        self._b += self._db_cache

    # 更新w,b
    def _update_params(self):
        self._w = self._alpha * self._y
        _idx = np.argmax((self._alpha != 0) & (self._alpha != self._c))
        # 要求下标满足0<alpha<C   & (self._alpha > 0) & (self._alpha < self._c)
        self._b = self._y[_idx] - np.sum(self._alpha * self._y * self._gram[_idx])  # y - y_hat

    def _update_alpha(self, idx1, idx2):
        # 更新α1，2
        # l, h存放α2对于的下界和上界。l1, h1存放α1的下界和上界
        # e1, e2存放两样本误差E1，E2
        # alpha1, alpha2存放更新后的α1，2
        # da1, da2存放Δα1，2.用于更新α1， dw, db， b1, b2
        y1, y2 = self._y[idx1], self._y[idx2]
        l2, h2 = self.get_lower_bound(idx1, idx2), self.get_upper_bound(idx1, idx2)
        h1, l1 = self._alpha[idx1] - y1 * y2 * (h2 - self._alpha[idx2]), self._alpha[idx1] - y1 * y2 * (
                l2 - self._alpha[idx2])
        k11, k12, k22 = self._gram[idx1][idx1], self._gram[idx1, idx2], self._gram[idx2][idx2]

        e1 = self._prediction_cache[idx1] - self._y[idx1]
        e2 = self._prediction_cache[idx2] - self._y[idx2]

        eta = k11 + k22 - 2*k12
        # 对于η值需要分情况讨论，尤其是在η=0, <0时
        if eta <= 0:
            lower_bound = self.optimizer_object(k11, k12, k22, y1, y2, l1, l2, e1, e2, self._b)
            upper_bound = self.optimizer_object(k11, k12, k22, y1, y2, h1, h2, e1, e2, self._b)
            if lower_bound > upper_bound:  # 取其中小者作为更新对象
                a2_new, a1_new = h2, h1
            else:
                a2_new, a1_new = l2, l1
        else:
            a2_new = self._alpha[idx2] - y2 * (e2 - e1) / eta
            a1_new = self._alpha[idx1] - y1 * y2 * (a2_new - self._alpha[idx2])

        da2 = - y2 * (e2 - e1) / eta  # a1,a2的变化量
        da1 = - y1 * y2 * (a2_new - self._alpha[idx2])
        # 约束取值范围
        if a2_new > h2:
            a2_new = h2
        elif a2_new < l2:
            a2_new = l2
        if a1_new > h1:
            a1_new = h1
        elif a1_new < l1:
            a1_new = l1
        #     更新a1, a2
        self._alpha[idx2] = a2_new
        self._alpha[idx1] = a1_new

        # 局部更新dw, db, prediction_cache
        self._update_dw_cache(idx1, idx2, da1, da2, y1, y2)
        self._update_db_cache(idx1, idx2, da1, da2, y1, y2, e1, e2)
        self._update_pred_cache(idx1, idx2)

    def _prepare(self, **kwargs):
        self._c = kwargs.get("c", KernelConfig.default_c)

    def _fit(self, sample_weight, tol):  # sample_weight,
        idx1 = self._pick_first(tol)
        # 若没能选出第一个变量， 则所有样本的误差都< theta, 此时返回真值，退出训练循环
        if idx1 is None:
            return True
        idx2 = self._pick_second(idx1)
        self._update_alpha(idx1, idx2)

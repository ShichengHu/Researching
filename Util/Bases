import numpy as np
import matplotlib.pyplot as plt


class ModelBase:
    def __init__(self, *args, **kwargs):
        self.name = self.__class__.__name__

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    def fit(self, *args, **kwargs):
        pass

    def predict(self, *arg, **kwargs):
        pass

    def evaluate(self, *args, **kwargs):
        pass

    def plot_acc_rate(self, *args, **kwargs):
        n_iter = kwargs.get("n_iter", 10*3)
        acc_lst = []
        best_index, worst_index = None, None
        best_acc, worst_acc = 0, 1
        for _ in range(n_iter):
            model = self.__class__()
            model.fit(*args[:2], **kwargs)  # work
            acc = model.evaluate(*args[2: 4], **kwargs)  # type:"float"
            # ["get_raw_result"] 不加acc它就认为是返回None
            acc_lst.append(acc)
            if acc < worst_acc:
                worst_acc = acc
                worst_index = _
            elif acc > best_acc:
                best_acc = acc
                best_index = _
        plt.plot(np.arange(n_iter), acc_lst, label="acc", lw=2.5)
        plt.plot([best_index, best_index, ], [0, best_acc, ], color="black", linestyle="--", lw=1.5)
        plt.plot([worst_index, worst_index, ], [0, worst_acc, ], color="cyan", linestyle="--", lw=1.5)
        plt.plot([0, n_iter, ], [np.average(acc_lst), np.average(acc_lst), ], color="grey", lw=2.5)
        plt.xlim(0, n_iter*1.1)
        plt.ylim(0, 1)
        plt.annotate(r"$best_i, best_a = {}, {}$".format(best_index, best_acc), xy=(best_index, best_acc),
                     xytext=(0.1*n_iter, 0.5), arrowprops=dict(facecolor="violet", arrowstyle='->',
                                                               connectionstyle="arc3,rad=.6"))
        plt.annotate(r"$worst_i, worst_a = {}, {}$".format(worst_index, worst_acc), xy=(worst_index, worst_acc),
                     xytext=(0.1 * n_iter, 0.3), arrowprops=dict(facecolor="violet", arrowstyle='->',
                                                                 connectionstyle="arc3,rad=.6"))
        plt.text(0.1*n_iter, 0.4, "average_acc {}".format(np.average(acc_lst)))
        plt.legend("upper right")
        plt.show()


class ClassifierBase(ModelBase):
    def __init__(self, *args, **kwargs):
        ModelBase.__init__(self)

    @staticmethod
    def acc(y, y_pred, weights=None, get_raw_result=False):
        y = np.array(y)
        y_pred = np.array(y_pred)
        return np.average(y == y_pred, weights=weights) if not get_raw_result \
            else "ACC is :{:8.6} %\n".format(100 * np.average(y == y_pred, weights=weights))

    def predict(self, x, **kwargs):  # get_raw_result=False
        pass

    def evaluate(self, x, y, get_raw_result=False):
        y = np.array(y)
        y_pred = self.predict(x)
        return np.average(y_pred == y) if not get_raw_result \
            else "Acc is : {:8.6} %".format(100*np.average(y_pred == y))


class KernelConfig:
    default_c = 1e0  # 1.0
    default_p = 3


class KernelBase(ClassifierBase):
    """
    _fit_args:
    _w :不再是np.sum(alpha*y*x)，而是alpha*y
    _b: np.sum(alpha*y)即np.sum(_w)
    因此predict值可以直接用_w*kernel+_b表示
    """

    def __init__(self):
        super(KernelBase, self).__init__()  # super(KernelBase).__init__()
        self._fit_args, self._fit_args_names = [], []
        self._x = self._y = None
        self._gram = None
        self._w = self._b = self._alpha = None
        self._kernel = self._kernel_param = None
        self._kernel_name = None
        self._prediction_cache = self._dw_cache = self._db_cache = None

    @staticmethod
    def _poly(x, y, p):
        # 定义多项式核函数，当p=1退化为线性核，高斯核也称RBF核,多加了1就有性能提升？
        return (x.dot(y.T) + 1) ** p  # 这里是否也会出现样本数不同的情况呢

    @staticmethod
    def _rbf(x, y, gamma):
        # rbf核
        # gama是高斯核的带宽,对于核函数，x的每一行需要乘y的每一行。
        return np.exp(- gamma*np.sum((x[:, None, :]-y)**2, axis=2))  # dot快还是升维快
        # return np.exp(-(x - y).dot(x - y) / (2 * gamma ** 2))  # 对于不同样本个数的情况没法计算

    @staticmethod
    def _linear(x, y):
        #     线性核
        return x.dot(y.T)

    # 默认使用rbf核
    def fit(self, x, y, kernel="rbf", epoch=10 ** 4, **kwargs):  #
        self._x, self._y = np.atleast_2d(x), np.atleast_1d(y)
        if kernel == "_poly":
            _p = kwargs.get("p", KernelConfig.default_p)
            self._kernel_name = "Polynomial"
            self._kernel_param = "degree = {}".format(_p)
            self._kernel = lambda _x, _y: KernelBase._poly(_x, _y, _p)
        elif kernel == "rbf":
            _gamma = kwargs.get("gamma", 1 / self._x.shape[1])  # 高斯核带宽定义为特征数的倒数
            self._kernel_name = "RBF"
            self._kernel_param = "gamma = {}".format(_gamma)
            self._kernel = lambda _x, _y: KernelBase._rbf(_x, _y, _gamma)
        elif kernel == "linear":
            self._kernel_name = "Linear"
            self._kernel = lambda _x, _y: KernelBase._linear(_x, _y)
        else:
            return ImportError("There is no kernel name : {}\n".format(kernel))
        # 初始化参数
        self._alpha = np.ones(len(self._x))
        self._prediction_cache = np.ones(len(self._x))
        self._w = np.zeros(len(self._x))
        self._b = 0.  # 0.
        self._gram = self._kernel(self._x, self._x)  #
        sample_weight = kwargs.get("sample_weight", None)  # 如果kwargs中有就得到这个参数，否则设为None
        #
        self._prepare(**kwargs)
        #
        _fit_args = []
        for _name, _arg in zip(self._fit_args_names, self._fit_args):
            if _name in kwargs:
                _arg = kwargs[_name]  # _arg = kwargs[_name]
                # 如果fit_args中有_arg那么为什么还有通过外界输入呢，很让人理解
                # 一方面_arg = _fit_args中元素，一方面还把_arg变成外接输入的值，这不是有病
                _fit_args.append(_arg)
                continue
            _fit_args.append(_arg)  # 如果外部输入中有,就从外部读取，否则读取内部给定值
        for _ in range(epoch):
            # if是不是意味着每次都执行呢？ yes
            if self._fit(sample_weight, _fit_args):  # sample_weight, *_fit_args   , _fit_args
                break
        self._update_params()  #

    def _update_pred_cache(self, *args):
        self._prediction_cache += self._db_cache
        if len(args) == 1:
            # self._prediction_cache += self._db_cache
            self._prediction_cache += self._dw_cache * self._gram[args[0]]
        else:
            # 之前用的是前两行，因为是部分更新
            # self._prediction_cache[args[0]] = np.sum(self._w * self._gram[args[0]]) + self._b
            # self._prediction_cache[args[1]] = np.sum(self._w * self._gram[args[1]]) + self._b
            # self._prediction_cache = self._w.dot(self._gram.T) + self._b  # equal
            self._prediction_cache += self._dw_cache.dot(self._gram[args, :])  # [args, :]

    def _prepare(self, **kwargs):
        pass

    def _fit(self, *args):  # *args, **kwargs
        pass

    def _update_db_cache(self, *args):
        pass

    def _update_dw_cache(self, *args):
        pass

    def _update_params(self):
        pass

    def predict(self, x, get_raw_result=False, gram_provided=False):
        # 计算测试集和训练集间的核矩阵，并用于决策,问题在于self._x和x的样本个数不见得同这样怎么减呢,升维可以
        if not gram_provided:
            x = self._kernel(np.atleast_2d(x), self._x)
        # y_pred = x.dot(self._w) + self._b  # self._w 是一维的dot就没有意义了,所以显然这一就不是一维的
        y_pred = np.sum(self._w[:, None] * x.T, axis=0)
        if get_raw_result:
            return y_pred
        else:
            return np.sign(y_pred)


class RegressionBase(ModelBase):
    def __init__(self, *arg, **kwargs):
        ModelBase.__init__(self)

    @staticmethod
    def acc(y, y_pred, get_raw_result=False):
        y = np.array(y)
        y_pred = np.array(y_pred)
        if get_raw_result:
            return np.average(y == y_pred)
        else:
            return "ACC is : {:8.6} %\n".format(100*np.average(y == y_pred))

    def evaluate(self, x, y, get_raw_result=False):
        x = np.atleast_2d(x)
        y = np.atleast_1d(y)
        y_pred = self.predict(x)
        return self.acc(y, y_pred, get_raw_result)

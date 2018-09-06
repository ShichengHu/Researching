# -*- coding:utf-8 -*-
import numpy as np
from copy import deepcopy  # res中的数据我不希望丢失，如果把res中的数据用deepcopy传给training_data就可以避免
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from time import time


class OperateData:

    @staticmethod
    # 归一化
    def normalize(x, trans_method='normal'):
        """
        :param x:
        :param trans_method: normal:归一化，适合于离散取值, standardize:标准化，适合于连续取值
        :return:
        upper bound: 上界,记录每个特征的上界
        lower bound:下界
        x_mean: 均值，每个特征的均值
        x_std: 标准差的似然估计，每个特征
        """
        x = np.atleast_2d(x)
        if trans_method == 'normal':
            upper_bound = np.max(x, axis=0)
            lower_bound = np.min(x, axis=0)
            return (x - lower_bound) / (upper_bound - lower_bound)
        elif trans_method == "standardize":
            x_mean = np.mean(x, axis=0)
            x_std = np.std(x, axis=0)
            return (x - x_mean)/x_std
        else:
            raise ImportError(": This trans_method {} isn't defined.\n".format(trans_method))

    @staticmethod
    def cv(fit, predict, x, y, k, shuffle=True):
        """
            交叉验证法
            :param fit: 训练函数
            :param predict: 预测函数
            :param x: 训练集数据
            :param y: 训练集数据的分类结果
            :param k: 交叉验证次数
            :param shuffle: 对数据打乱
            :return: k折交叉验证的平均预测结果，这里用到预测精度
            """
        sum_ = 0
        x, y = np.atleast_2d(x), y.reshape((-1, 1))
        # if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        z = np.hstack((x, y))
        if shuffle:
            np.random.shuffle(z)
        res = []
        step = int(len(x) / k)
        for i in range(1, k):
            res.append(z[(i - 1) * step: i * step])
        res.append(z[(k - 1) * step:])
        for i in range(k):
            train_data = deepcopy(res)
            validation_set = train_data.pop(i)
            training_set = train_data
            training_set = np.concatenate([arr for arr in training_set])
            x_t, y_t = np.hsplit(training_set, indices_or_sections=[x.shape[1]])
            x_v, y_v = np.hsplit(validation_set, indices_or_sections=[x.shape[1]])
            # 这里还需要执行这两个函数，而且我还没法直接调用acc,否则返回sum_就是一个矩阵，不明白为啥
            fit(x_t, y_t)
            y_p = predict(x_v)
            acc = np.sum(y_p == y_v) / len(x_v)  # 定义了评估函数，这里我用的是精度
            sum_ += acc
        mse = sum_ / k
        return mse


# 测试银行所用的处理
def get_dataset(name, path, train_num=None, tar_idx=-1, shuffle=False):
    x = []
    with open(path) as f:
        if 'bank' or 'data' in name:
            for sample in f:
                sample = sample.replace('"', '')
                x.append(sample.strip().split(';'))
    if shuffle:
        np.random.shuffle(x)
    y = np.array([xi.pop(tar_idx) for xi in x])
    x = np.array(x)
    if train_num is None:
        return x, y
    return x[:train_num], y[:train_num], x[train_num:], y[train_num:]


# 测试ballon,mushroom专用的数据预处理
def get_dataset_(name, path, train_num=None, tar_idx=-1, shuffle=False):
    x = []
    with open(path) as file:
        if 'ballon' or 'mushroom' in name:
            for sample in file:
                x.append(sample.strip().split(","))
    if shuffle:
        np.random.shuffle(x)
    y = np.array([xx.pop(tar_idx) for xx in x])
    x = np.array(x)
    if train_num is None:
        return x, y
    return x[:train_num], y[:train_num], x[train_num:], y[train_num:]


# 对读取分割完成的数据所作处理
def quantize_data(x, y, wc=None, continuous_rate=0.1, shuffle=True, separate=False):
    """
    :param x: 记录预处理后的数据
    :param y: 记录预处理后的类别
    :param wc: 记录各属性的连续性
    :param separate:
    :param shuffle:
    :param continuous_rate: 记录判断连续性参数
    :return:
    label_dict: 记录从数值化前的类别值与数值化后直接的对应关系
    feat_dics: 记录数值化后与数值化前数据元素的对应关系字典
    """
    # 对列表，矩阵转置
    if isinstance(x, list):
        xt = map(list, zip(*x))
    else:
        xt = x.T
    indices = np.random.permutation(len(y))
    x, y = x[indices, :], y[indices]
    features = [set(feat) for feat in xt]
    if wc is None:
        wc = np.array([len(feat) >= int(continuous_rate * len(y)) for feat in features])
    else:
        wc = np.asarray(wc)
    feat_dicts = [
        {val: key for key, val in enumerate(feats)} if not wc[idx] else {None}
        for idx, feats in enumerate(features)
    ]
    # 怎么理解separate呢?把矩阵根据特征划分为连续阵和离散阵
    # if np.all(~wc):  # 如果wc的互补矩阵的所有元素都连续(wc都离散)
    #     data_type = np.int  # 设置元素类型为int(np或可以去掉)有连续值时可能取浮点数，贸然设置为int会改变数据
    # else:
    #     data_type = np.float32  # float32
    if not separate:
        if np.all(~wc):  # 所有特征都不连续， 对数值化结果取整
            data_type = np.int
        else:
            data_type = np.float32  # 有连续特征
        x = np.array([[feat_dicts[i][_l] if not wc[i] else _l for i, _l in enumerate(sample)]
                      for sample in x], dtype=data_type)
    else:
        x = np.array([[feat_dicts[i][_l] if not wc[i] else _l for i, _l in enumerate(sample)]
                      for sample in x], dtype=np.float32)
        x = (x[:, ~wc].astype(np.int), x[:, wc])  # 离散特征就取整， 连续取浮点
    # dtype = None
    # 如果维度取值是连续的那么数值化x只需要对离散取值的变量处理，否则直接把原始值记录
    # x = np.array([
    #     [feat_dicts[idx][val] if not wc[idx] else val for idx, val in enumerate(sample)]for sample in x],
    # dtype=data_type)
    label_dict = {v: k for k, v in enumerate(set(y))}
    y = np.array([label_dict[yy] for yy in y], dtype=np.int8)
    # print(feat_dicts)
    label_dict = {val: label for label, val in label_dict.items()}  # 有时候有用，有时候无用？
    return x, y, wc, features, feat_dicts, label_dict


def data_numeral(name=None, path=None, shuffle=True, wc=None, tar=-1, only_train=True, transform=True,
                 continuous_ratio=0.25, train_ratio=0.8):
    # 决策树
    # transform: x需要数值化
    # name: 文件名
    # path: 路径
    # shuffle: 打乱顺序
    # wc: 各维度连续性
    # tar: y对应维度
    # only_train: 不分割为训练集和测试集
    # continuous_ratio: 连续率
    # train_ratio: 训练集占比
    _data, _x, _y = [], [], []
    with open(path) as file:
        for line in file:
            line = line.replace('\t', ',') if name == "water_melon" else line.replace('?', ' ')
            _data.append(line.strip().split(','))
    if shuffle:
        np.random.shuffle(_data)

    for line in _data:
        _y.append(line.pop(tar))
        _x.append(line)
    _x = np.array(_x)
    _y = np.array(_y)

    m, n = _x.shape
    features = [set(xx) for xx in _x.T]
    if wc is None:
        wc = np.array([len(feat) > continuous_ratio * len(_y) for feat in features])
    feat_dic = [{v: k for k, v in enumerate(feature)} if not wc[idx] else None for idx, feature in enumerate(features)]
    rev_feat_dic = [{k: v for v, k in feature.items()} if not wc[idx] else None for idx, feature in enumerate(feat_dic)]
    label_dic = {v: k for k, v in enumerate(set(_y))}
    label_dic = {k: v for v, k in label_dic.items()}

    rs_x = [[] for _ in range(n)]
    for idx in range(n):
        xx = _x.T[idx]
        if wc[idx]:
            rs_x[idx] = [float(val) for val in xx]
        else:
            rs_x[idx] = [feat_dic[idx][val] for val in xx]
    x_trans = np.array(rs_x).T
    # _y = np.array(label_dic[val] for val in _y)
    if transform:
        if only_train:
            return x_trans, _y, label_dic, rev_feat_dic, wc
        else:
            train_num = int(train_ratio*len(x_trans))
            x_train = x_trans[: train_num]
            y_train = _y[: train_num]
            x_test = x_trans[train_num:]
            y_test = _y[train_num:]
            return x_train, y_train, x_test, y_test, label_dic, rev_feat_dic, wc
    else:
        if only_train:
            return _x, _y, label_dic, rev_feat_dic, wc
        else:
            train_num = int(train_ratio*len(_x))
            x_train = _x[: train_num]
            y_train = _y[: train_num]
            x_test = _x[train_num:]
            y_test = _y[train_num:]
            return x_train, y_train, x_test, y_test, label_dic, rev_feat_dic, wc


def train_test_split(x, y, test_ratio=0.2, shuffle=True, seed=None):
    """
    logit_regression
    :param x:
    :param y:
    :param test_ratio:
    :param shuffle:
    :param seed:
    :return:
    """
    assert len(x) == len(y), "长度同"
    x = np.atleast_2d(x)
    y = np.atleast_1d(y)
    z = np.hstack((x, y.reshape(-1, 1)))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(z)
    x = z[:, : - 1]
    y = z[:, -1]
    test_indices = int(len(x)*test_ratio)
    return x[test_indices:], y[test_indices:], x[:test_indices], y[:test_indices]


# ============================= 生成类别空间为1，-1的只含两个特征的数据集 ========================
# 生成随机数据集
def gen_random(size=100):
    xy = np.random.rand(size, 2)
    z = np.random.randint(2, size=size)
    z[z == 0] = -1  # Adaboost框架要求类别空间是-1，1
    return xy, z


# 生成异或数据集
def gen_xor(size=100):
    x = np.random.randn(size)
    y = np.random.randn(size)
    z = np.ones(size)
    z[x * y > 0] = -1
    return np.c_[x, y].astype(np.float32), z


# 生成螺旋线数据集
def gen_spiral(size=50, n=4, scale=2):
    xs = np.zeros((size * n, 2), dtype=np.float32)
    ys = np.zeros(size * n, dtype=np.int8)
    for i in range(n):
        ix = range(size * i, size * (i + 1))
        r = np.linspace(0.0, 1, size+1)[1:]
        t = np.linspace(2 * i * np.pi / n, 2 * (i + scale) * np.pi / n, size) + np.random.random(size=size) * 0.1
        xs[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        ys[ix] = 2 * (i % 2) - 1  # ys的取值只能是1或者-1
    return xs, ys


# 画出决策边界；如果只关心算法本身，可以略去这一段代码不看
def visualize2d(clf, x, y, draw_background=False):
    """
    绘制决策边界，要求x得是二维的
    :param clf: 分类器
    :param x: 测试集
    :param y:
    :param draw_background:b背景
    :return:
    """
    axis, labels = np.array(x).T, np.array(y)
    decision_function = lambda xx: clf.predict(xx)  # xx只能是二维的，这点可还不够

    nx, ny, padding = 400, 400, 0.2  # x轴，y轴取值点数
    x_min, x_max = np.min(axis[0]), np.max(axis[0])  # 限制x轴取值范围
    y_min, y_max = np.min(axis[1]), np.max(axis[1])  # 限制y轴取值范围
    x_padding = max(abs(x_min), abs(x_max)) * padding  # 边缘空隙
    y_padding = max(abs(y_min), abs(y_max)) * padding
    x_min -= x_padding  # 修正范围
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding

    def get_base(xx, yy):
        _xf = np.linspace(x_min, x_max, xx)
        _yf = np.linspace(y_min, y_max, yy)

        n_xf, n_yf = np.meshgrid(_xf, _yf)
        return _xf, _yf, np.c_[n_xf.ravel(), n_yf.ravel()]

    xf, yf, base_matrix = get_base(nx, ny)
    z = decision_function(base_matrix).reshape((nx, ny))

    labels[labels == -1] = 0  # SVM中判为-1的化为0
    xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)

    plt.figure()
    if draw_background:
        plt.pcolormesh(xy_xf, xy_yf, z, cmap=colors.ListedColormap(["#EF9A9A", "#F5FFFA", '#00FFFF']))
        # Paired, RdYlBu  plt.cm.RdYlBu
    else:
        plt.contour(xf, yf, z, c='k-', levels=[0])
    plt.scatter(axis[0], axis[1], c=y, cmap=colors.ListedColormap(["r", "y", "b"]))  # colors
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()
# =====================================================================================


# WiFi定位数据预处理
class Wifi:

    @staticmethod
    def project_get_dataset(path=None, split=True, train_num=None, train_ratio=0.8, tar_idx=-1,
                            shuffle=False, normalize=False, trans_method=None, svm=False):
        """
        数据预处理

        :param path: 路径
        :param split: 分割数据集
        :param train_num: 训练集数目
        :param train_ratio: 训练率
        :param tar_idx: 输出类别维度
        :param shuffle: 打乱顺序
        :param normalize: 归一化
        :param trans_method: 归一化方法
        :param svm: 应用SVM算法
        :return: 处理后的结果
        """
        x = []
        with open(path) as f:
            for sample in f:
                x.append(sample.strip().split('\t'))
        if shuffle:
            np.random.shuffle(x)
        y = np.array([xi.pop(tar_idx) for xi in x], dtype=int)
        if svm:
            p1 = y == 0
            y[p1] = -1
        x = np.array(x, dtype=int)
        if normalize:
            x = OperateData.normalize(x, trans_method=trans_method)
        if not split:
            return x, y
        else:
            if train_num is None:
                train_num = int(len(x) * train_ratio)
                return x[:train_num], y[:train_num], x[train_num:], y[train_num:]
            else:
                return x[:train_num], y[:train_num], x[train_num:], y[train_num:]

    @staticmethod
    def view(model, x_train, y_train, x_test, y_test):
        """
        可视化结果

        :param model: 调用的模型， 用于对输入数据处理得到预测结果
        :param x_train: 训练数据
        :param y_train:
        :param x_test: 测试数据
        :param y_test:
        :return: 三维图
        """
        if not isinstance(x_train, np.ndarray):
            raise ImportError("data must be np.ndarray")
        assert len(x_train) == len(y_train) and len(x_test) == len(y_test), "Length must be equal."
        assert x_train.shape[1] == 3, "wifi测试数据有三个维度。"
        y_pred = model.predict(x_test)
        tmp_label = y_pred == y_test
        mask = [tmp_label, ~tmp_label]
        fig = plt.figure()
        ax = Axes3D(fig)
        # ax = plt.subplot(111, projection='3d')  # 创建三维图
        ax.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], x_train[y_train == 0, 2], color="LightSkyBlue",
                   marker="o", label="train_data0")
        ax.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], x_train[y_train == 1, 2], color="orange",
                   marker="o", label="train_data1")
        ax.scatter(x_test[tmp_label][y_test[tmp_label] == 0, 0], x_test[tmp_label][y_test[tmp_label] == 0, 1],
                   x_test[tmp_label][y_test[tmp_label] == 0, 2], color="LightSkyBlue", marker='^',
                   label="not be affected test_data0")
        ax.scatter(x_test[tmp_label][y_test[tmp_label] == 1, 0], x_test[tmp_label][y_test[tmp_label] == 1, 1],
                   x_test[tmp_label][y_test[tmp_label] == 1, 2], color="orange", marker='^',
                   label="not be affected test_data1")
        ax.scatter(x_test[mask[1]][y_test[mask[1]] == 0, 0], x_test[mask[1]][y_test[mask[1]] == 0, 1],
                   x_test[mask[1]][y_test[mask[1]] == 0, 2], color="cyan", marker="*", label="changed test_data0")
        ax.scatter(x_test[mask[1]][y_test[mask[1]] == 0, 0], x_test[mask[1]][y_test[mask[1]] == 0, 1],
                   x_test[mask[1]][y_test[mask[1]] == 0, 2], color="violet", marker="*", label="changed test_data1")
        ax.set_xlabel('0')
        ax.set_ylabel('1')
        ax.set_zlabel('2')
        plt.legend()
        plt.savefig("data_distributation.jpg")
        plt.show()

    def draw_precision_map(self, model, test_set=None, n_iter=1000, **kwargs):
        set1 = kwargs.get("0228", "C://Users//C//Desktop//3ap//0228.txt")
        set2 = "C://Users//C//Desktop//3ap//{}.data".format(test_set)
        acc_lst = []
        best_index, worst_index = None, None
        best_acc, worst_acc = 0, 1
        t = time()
        for _ in range(n_iter):
            print(_)
            np.random.seed(_)
            if test_set is None:
                x_train, y_train, x_test, y_test = \
                    self.project_get_dataset(path=set1, normalize=True, trans_method="standardize",
                                             svm=kwargs.get("svm", False))
            else:
                x_train, y_train = self.project_get_dataset(path=set1, split=False, normalize=True,
                                                            trans_method="standardize", svm=kwargs.get("svm", False))
                x_test, y_test = self.project_get_dataset(path=set2, split=False, normalize=True,
                                                          trans_method="standardize", svm=kwargs.get("svm", False))
            # x_train, y_train = x_train[: int(len(x_train) * 0.90)], y_train[: int(len(x_train) * 0.90)]
            # x_test, y_test = x_test[:int(len(x_test) * 0.90)], y_test[: int(len(x_test) * 0.90)]
            built_model = model()
            built_model.fit(x_train, y_train, **kwargs)
            y_pred = built_model.predict(x_test, **kwargs)
            acc = np.average(y_pred == y_test)
            acc_lst.append(acc)
            if acc < worst_acc:
                worst_acc = acc
                worst_index = _
            elif acc > best_acc:
                best_acc = acc
                best_index = _

        average_time = (time() - t) / n_iter
        plt.plot(np.arange(n_iter), acc_lst, label="accuracy")  # , lw=2.5
        plt.plot([best_index, best_index, ], [0, best_acc, ], "k--")  # color="black", linestyle="--"
        plt.plot([worst_index, worst_index, ], [0, worst_acc, ], linestyle="--")  # color="cyan", linestyle="--"
        plt.plot([0, n_iter, ], [np.average(acc_lst), np.average(acc_lst), ], color="grey")
        plt.ylim(0, 1)
        plt.annotate(r"$best_i, best_a = {}, {}$".format(best_index, best_acc), xy=(best_index, best_acc),
                     xytext=(0.1 * n_iter, 0.5), arrowprops=dict(arrowstyle='->', facecolor="violet",
                                                                 connectionstyle="arc3,rad=.6"))
        plt.annotate(r"$worst_i, worst_a = {}, {}$".format(worst_index, worst_acc), xy=(worst_index, worst_acc),
                     xytext=(0.1 * n_iter, 0.3), arrowprops=dict(arrowstyle='->', facecolor="violet",
                                                                 connectionstyle="arc3,rad=.6"))  # ,
        plt.text(0.1 * n_iter, 0.4, "average_acc {}, average_time {}".format(np.average(acc_lst), average_time))
        plt.title("{}_{}_{}".format(model.__name__, "0228", test_set))
        plt.legend("upper right")
        plt.show()

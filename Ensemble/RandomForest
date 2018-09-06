from DecisionTree.CvDTree import *
from Util.Bases import ClassifierBase
import numpy as np


class RandomForest(ClassifierBase):
    """
    随机森林算法：

    运用Bootstrap法获取n个样本，可以认为这样获得数据服从同一分布，这样得到一个数据集
    对数据集用不带后剪枝的决策树生成，一词得到多棵树（对于树中的每个节点需要采用随机抽取的方式从可选特征中选择部分特征作
    为可选特征，目的是使得结果变化更明显，每棵树由此可能长得不一样）
    预测结果是选择投票这一简单方式，票数多者为结果

    随机森林是一种典型的集成学习方法（Bootstrap Aggregating), 弱分类器集成强分类器
    """

    _cvd_tree = {
        "id3": ID3Tree,
        "c45": C45Tree,
        "cart": CartTree
    }

    def __init__(self):
        super(RandomForest, self).__init__()
        self.trees = []

    # 对每个样本进行投票决策
    @staticmethod
    def most_appearance(arr):
        u, c = np.unique(arr, return_counts=True)  # np.unique()返回的结果可不是一定从0开始
        return u[np.argmax(c)]

    def fit(self, x, y, sample_weight=None, tree="cart", epoch=1000, feature_bound="log", *args, **kwargs):  # *args,
        x, y = np.atleast_2d(x), np.atleast_1d(y)
        n_sample = len(y)
        for _ in range(epoch):
            print(_)
            _indices = np.random.randint(n_sample, size=n_sample)  # 采用Bootstrap方式重复n次放回式抽样，获取n个数据
            if sample_weight is None:
                _local_weight = None
            else:
                _local_weight = sample_weight[_indices]
                _local_weight /= np.sum(_local_weight)
            tmp_tree = RandomForest._cvd_tree[tree](*args, **kwargs)  # *args,
            tmp_tree.fit(x[_indices], y[_indices], sample_weight=_local_weight, feature_bound=feature_bound, rf=True)
            self.trees.append(deepcopy(tmp_tree))

    # 对个体决策简单组合
    def predict(self, x, **kwargs):
        x = np.array(x)
        _matrix = np.array([_tree.predict(x) for _tree in self.trees]).T
        return np.array([self.most_appearance(rs) for rs in _matrix])

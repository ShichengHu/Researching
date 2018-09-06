from Bayes.Vectorized.GaussianNB import GaussianNB
from DecisionTree.CvDTree import *
from Bayes.Vectorized.MultinomialNB import MultinomialNB
from SVM.SVM import SVM
from LogisticRegression.LogisticRegression import LogisticRegression
from sklearn.svm import SVC
from sklearn import naive_bayes, tree
import numpy as np


class AdaBoost:
    # 弱分类器字典
    _weak_clf = {
        "SKNB": naive_bayes.MultinomialNB,
        "SKGNB": naive_bayes.GaussianNB,
        "NB": MultinomialNB,
        "GNB": GaussianNB,
        "Cart": CartTree,
        "C45": C45Tree,
        "ID3": ID3Tree,
        "DT": tree.DecisionTreeClassifier,
        "SVC": SVC,
        "SVM": SVM,
        "LR": LogisticRegression
    }

    def __init__(self):
        self._clf, self._clfs, self._clfs_weights = "", [], []

    def fit(self, x, y, sample_weight=None, clf=None, epoch=10, eps=1e-12, **kwargs):
        x, y = np.atleast_2d(x), np.atleast_1d(y)
        if clf is None or AdaBoost._weak_clf[clf] is None:
            clf = "Cart"
            kwargs = {"max_depth": 1, "whether_continuous": np.array([True, True, True])}
            # kwargs["max_depth"] = 1
        self._clf = clf
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        sample_weight /= len(y)
        for _ in range(epoch):
            tmp_clf = AdaBoost._weak_clf[clf](**kwargs)
            tmp_clf.fit(x, y, sample_weight=sample_weight)  # , **kwargs
            y_pred = tmp_clf.predict(x)  # model prediction
            mask = y_pred == y
            # tmp_weight = np.ones(len(y))
            # tmp_weight[mask] = 0
            # 加权错误率, 用eps增加稳定性
            em = min(max((~mask).dot(sample_weight[:, None])[0], eps), 1-eps)  # True * 1 = 1
            # 点乘结果是只含一个元素的矩阵
            # equal to min(max(np.sum((~mask)*sample_weight), eps), 1-eps)
            # em = np.sum(tmp_weight*sample_weight)  # type: # float
            # em = min(max(eps, em), 1-eps)
            # 话语权
            am = 0.5 * np.log(1/em - 1)
            sample_weight *= np.exp(-am*y*y_pred)  # 选择的损失函数是指数函数
            sample_weight /= np.sum(sample_weight)
            # 更新并记录用的模型和这个模型在决策中的作用
            self._clfs.append(deepcopy(tmp_clf))
            self._clfs_weights.append(am)

    def predict(self, x):
        x = np.atleast_2d(x)
        rs = np.zeros(len(x))
        for clf, am in zip(self._clfs, self._clfs_weights):
            pr = clf.predict(x)
            rs += am*pr
        return np.sign(rs)  # np.round(rs)

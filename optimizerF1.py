from functools import partial
import numpy as np
import scipy as sp
from sklearn.metrics import f1_score


def row_multi_f1_score(y_true, y_pred, weights_=None, average='macro'):
    """
    根据 weights 计算 f1分数
    :param y_true: shape(n,)
    :param y_pred: shape(n,num_class)
    :param weights_: weights of each class
    :return: f1_score(y_true, np.argmax(y_pred*weights))
    """
    if weights_ is None:
        print("Warning weight is None")
        weights_ = np.array([1] * y_pred.shape[1])

    assert len(weights_) == y_pred.shape[1], "the shape of weight != the shape of y_pred"
    # print(y_pred)
    return f1_score(
        y_true=y_true, y_pred=np.argmax(y_pred * weights_, axis=1), average=average
    )


class MultiF1Optimizer:
    def __init__(self, loss_fn=None, num_classes=25):
        self.coef_ = {}
        self.loss_fn = loss_fn if loss_fn is not None else row_multi_f1_score
        self.coef_["x"] = np.array([1] * num_classes)
        self.num_class = num_classes

    def _loss(self, coef, X, y):
        ll = self.loss_fn(y, X, coef)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._loss, X=X, y=y)
        initial_coef = np.array([1] * self.num_class)
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method="nelder-mead"
        )

    def coefficients(self):
        return self.coef_["x"]

    def calc_score(self, X, y, coef=None):
        return self.loss_fn(y, X, coef)


if __name__ == '__main__':
    thresholder = MultiF1Optimizer()

    y_true = np.random.randint(0, 5, (100,))
    y_pred = np.random.randn(100, 6)
    print(y_pred.shape, y_true.shape)
    print("原始f1 ", thresholder.calc_score(y_pred, y_true))

    thresholder.fit(y_pred, y_true)
    coef = thresholder.coefficients() # 稀疏
    f1_score = thresholder.calc_score(y_pred, y_true, coef)
    print(coef, "优化后的f1 分数", f1_score)
    # print("优化后的logits：", y_pred * coef)

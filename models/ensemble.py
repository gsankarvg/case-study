import numpy as np

def compute_weights(nrmse1, nrmse2):
    inv1 = 1 / nrmse1
    inv2 = 1 / nrmse2

    w1 = inv1 / (inv1 + inv2)
    w2 = inv2 / (inv1 + inv2)

    return w1, w2


def ensemble_predict(model1, model2, X, w1, w2):
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)

    return w1 * pred1 + w2 * pred2

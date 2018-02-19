import sys

import numpy as np
from scipy import io 

def load_single_sub(sub, cut):
    snic_tmp = "C:/Users/Albin Heimerson/Desktop/exjobb/"
    if len(sys.argv) > 1:
        snic_tmp = str(sys.argv[1])
    xn = None
    yn = None
    names = ["FA", "LM", "OB"]
    for i in range(3):
        name = "Subj{:02}_CleanData_study_{}".format(sub, names[i])
        print("loading ", name)
        m = io.loadmat('{}/DATA/Visual/{}.mat'.format(snic_tmp, name))
        trials = m[name][0][0][2][0]
        if cut:
            for j in range(trials.shape[0]):
                trials[j] = trials[j][:, 768:1536]
        labels = np.zeros((trials.shape[0], 3))
        labels[:, i] = 1
        if xn is None:
            xn = trials
            yn = labels
        else:
            xn = np.concatenate((xn, trials), axis=0)
            yn = np.concatenate((yn, labels), axis=0)

    xn = np.stack(xn, axis=0)
    n = xn.shape[0]
    s = np.arange(n)
    np.random.shuffle(s)

    return (xn[s], yn[s])


def load_single(idx=None, cut=True):
    x = []
    y = []

    if idx is None:
        for sub in [i if i < 10 else i + 1 for i in range(1, 19)]:  # 19 is max
            xn, yn = load_single_sub(sub, cut)
            x.append(xn)
            y.append(yn)
    else:
        x, y = load_single_sub(idx, cut)

    return (x, y)


def load_all(cut=True):
    x = None
    y = None

    for sub in [i if i < 10 else i + 1 for i in range(1, 19)]:  # 19 is max
        xn, yn = load_single_sub(sub, cut)
        if x is None:
            x = xn
            y = yn
        else:
            x = np.concatenate((x, x), axis=0)
            y = np.concatenate((y, y), axis=0)

    print(x.shape, y.shape)
    x = np.stack(x, axis=0)
    print(x.shape)
    s = np.arange(x.shape[0])
    np.random.shuffle(s)

    return (x[s], y[s])

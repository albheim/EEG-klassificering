import sys

import numpy as np
from scipy import io

def load_single_sub(sub, cut=True, study=True, shuffle=True, visual=True, transpose=False):
    snic_tmp = "C:/Users/Albin Heimerson/Desktop/exjobb"
    if len(sys.argv) > 1:
        snic_tmp = str(sys.argv[1])
    xn = None
    yn = None
    names = ["FA", "LM", "OB"]
    for i in range(3):
        name = "Subj{:02}_CleanData_{}_{}".format(sub,
                                                  'study' if study else 'test',
                                                  names[i])
        if not study:
            name += "_{}".format("visual" if visual else "lexical")

        print("loading: ", name)
        m = io.loadmat('{}/DATA/{}/{}.mat'.format(snic_tmp,
                                                  "Visual" if visual else "Verbal",
                                                  name))
        trials = m[name][0][0][2][0]
        if cut:
            for j in range(trials.shape[0]):
                trials[j] = trials[j][:, 768:1536]
                if transpose:
                    trials[j] = trials[j].T
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
    if shuffle:
        np.random.shuffle(s)

    return (xn[s], yn[s])


def load_single(idx=None, cut=True, shuffle=True, visual=True, transpose=False,
                study=True):
    x = []
    y = []

    if visual:
        subs = [i if i < 10 else i + 1 for i in range(1, 19)]
    else:
        subs = [1, 2, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22]

    if idx is None:
        for sub in subs:
            xn, yn = load_single_sub(sub, cut, study, shuffle, visual, transpose)
            x.append(xn)
            y.append(yn)
    else:
        x, y = load_single_sub(idx, cut, study, shuffle, visual, transpose)

    print(x[0].shape)
    return (x, y)


def load_all(cut=True, visual=True):
    x = None
    y = None

    if visual:
        subs = [i if i < 10 else i + 1 for i in range(1, 19)]
    else:
        subs = [1, 2, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22]

    for sub in subs:  # 19 is max
        xn, yn = load_single_sub(sub, cut, visual)
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


def load_marg(cut=None, visual=True, shuffle=True):
    snic_tmp = "C:/Users/Albin Heimerson/Desktop/exjobb"
    if len(sys.argv) > 1:
        snic_tmp = str(sys.argv[1])
    x = []
    y = []
    if visual:
        subs = [i if i < 10 else i + 1 for i in range(1, 19)]
    else:
        subs = [1, 2, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    for sub in subs:
        xn = None
        yn = None
        for i in range(3):
            name = "Subj{:02}_{}_marg".format(sub, "Visual" if visual else "Verbal")
            print("loading: ", name)
            m = io.loadmat('{}/DATA/Modified/marginal/{}.mat'.format(snic_tmp, name))

            trials = m['data'][0][0][i][:, 0]
            if cut is not None:
                for j in range(trials.shape[0]):
                    trials[j] = trials[j][:, cut[0]:cut[1]]
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
        if shuffle:
            np.random.shuffle(s)
        x.append(xn[s])
        y.append(yn[s])

    print(x[0].shape)
    return (x, y)


def cut(data, cut=[768, 1536], tcut = None):
    if tcut is not None:
        cut[0] = int(512 * tcut[0] + 768)
        cut[1] = int(512 * tcut[1] + 768)
    print(cut)
    for sub in range(len(data)):
        data[sub] = data[sub][:, :, cut[0]:cut[1]]


def modify(x, y, n, nmult=0, displacement=0, cut=[768, 1536]):
    mdata = [None for i in range(len(x))]
    my = [None for i in range(len(x))]
    for sub in range(len(mdata)):
        s = x[sub].shape[0]
        my[sub] = np.zeros((s * n, 3))
        for j in range(0, s * n, s):
            my[sub][j:j + s] = y[sub]
        mdata[sub] = np.zeros((s * n, x[sub].shape[1], cut[1] - cut[0]))
        mdata[sub][:s, :, :] = x[sub][:, :, cut[0]:cut[1]]
        for j in range(s, s * n, s):
            for i in range(len(x[sub])):
                if displacement > 0:
                    d = np.random.randint(-displacement, displacement + 1)
                    mdata[sub][j:j + s, :, :] = x[sub][:, :, cut[0] + d:cut[1] + d]

                if nmult != 0:
                    for k in range(mdata[sub][i].shape[0]):
                        mdata[sub][j + i, k] += nmult * np.std(mdata[sub][i, k]) * np.random.ranf(mdata[sub][i, k].shape)

    return mdata, my


def load_spect_downsample(visual=True, shuffle=True, ds=8):
    snic_tmp = "C:/Users/Albin Heimerson/Desktop/exjobb"
    if len(sys.argv) > 1:
        snic_tmp = str(sys.argv[1])
    x = []
    y = []
    if visual:
        subs = [i if i < 10 else i + 1 for i in range(2, 19)] # change to 1
    else:
        subs = [1, 2, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    for sub in subs:
        xn = None
        yn = None
        for i in range(3):
            name = "Subj{:02}_{}_spec{}".format(sub, "Visual" if visual else "Verbal", ds)
            print("loading: ", name)
            m = io.loadmat('{}/DATA/Modified/spect_downsample/{}.mat'.format(snic_tmp, name))

            trials = m['data'][0][0][i][:, 0]
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
        if shuffle:
            np.random.shuffle(s)
        x.append(xn[s])
        y.append(yn[s])

    print(x[0].shape)
    return (x, y)

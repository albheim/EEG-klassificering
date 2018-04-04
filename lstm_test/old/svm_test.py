import numpy as np
import numpy as np

import data
import util

from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing

x, y = data.load_single(cut=True, visual=True, transpose=False)
xt, yt = data.load_single(cut=True, visual=True, study=False, transpose=False)
print(x[0].shape)


splits = 10
n_subs = len(x)
n_models = 1
accs = [0 for j in range(n_models)]
accs2 = [0 for j in range(n_models)]


channels = [4, 23]  # 4, 14, 23
ds = 8
for i in range(n_subs):
    s, _, l = x[i].shape
    c = len(channels)
    x[i] = x[i][:, channels, :]
    x[i] = x[i][:, :, range(0, l, ds)].reshape((s, l * c // ds))
    # x[i] = preprocessing.scale(x[i])
    y[i] = np.argmax(y[i], axis=1)
    s, _, l = xt[i].shape
    xt[i] = xt[i][:, channels, :]
    xt[i] = xt[i][:, :, range(0, l, ds)].reshape((s, l * c // ds))
    # xt[i] = preprocessing.scale(xt[i])
    yt[i] = np.argmax(yt[i], axis=1)


for j in range(n_models):
    avgacc = 0
    avgacc2 = 0
    for i in range(n_subs):
        n = x[i].shape[0]
        acc = 0
        acc2 = 0
        for tr, val in util.kfold(n, splits, shuffle=True):
            model = SVC(kernel='linear')

            # fit with next kfold data
            model.fit(x[i][tr], y[i][tr])

            acc += np.mean(y[i][val] == model.predict(x[i][val]))
            acc2 += np.mean(yt[i] == model.predict(xt[i]))

        acc /= splits
        acc2 /= splits
        avgacc += acc
        avgacc2 += acc2

        print("subject {}, avg accuracy {}/{} over {} splits".format(i + 1 if i + 1 < 10 else i + 2,
                                                                     acc, acc2, splits))

    avgacc /= n_subs
    accs[j] = avgacc
    avgacc2 /= n_subs
    accs2[j] = avgacc2
    print("avg accuracy over all subjects {}/{}".format(avgacc, avgacc2))


for a, a2 in sorted(zip(accs, accs2)):
    print("acc {}/{}\n".format(a, a2))

print("avg over all trials and subjects {}/{}".format(sum(accs) / len(accs), sum(accs2) / len(accs2)))

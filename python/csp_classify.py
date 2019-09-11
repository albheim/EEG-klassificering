import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import create_info
from mne.channels import read_layout
from csp import CSP

import data
import util

tmin, tmax = -1.5, 2.5
event_id = dict(FA=0, LM=1, OB=2)

acc = 0

bin_size = 20
n_bins = 5
p_size = bin_size * n_bins

start = 700
end = 1300
last = end - p_size
timepoints = range(start, last, bin_size)
steps = len(timepoints)

heatmap = np.zeros((steps, steps))

info = create_info(
    ch_names=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FCz',
            'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2',
            'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'O2', 'PO10', 'Iz'],
    ch_types=['eeg' for _ in range(31)],
    sfreq=512
)

# Assemble a classifier
lda = LinearDiscriminantAnalysis()
# qda = QuadraticDiscriminantAnalysis()
mlp = MLPClassifier(activation='tanh', solver='adam', alpha=1e-5,
                    learning_rate='invscaling', verbose=False, max_iter=500,
                    hidden_layer_sizes=(20, 10, 3), random_state=1)
svc = SVC(kernel='linear')
csp = CSP(n_components=8, reg=None, log=True, norm_trace=False)

x, y = data.load_single(cut=True)
y = [np.where(yi==1)[1] for yi in y]
xt, yt = data.load_single(cut=True, study=False)
yt = [np.where(yi==1)[1] for yi in yt]

subs = len(x)
splits = 10
for name, classifier in [('LDA', lda),
                         ('SVM', svc),
                         ('MLP', mlp)]:
    acc = 0
    acc2 = 0
    for sub in range(18):
        n = x[sub].shape[0]

        for tr, val in util.kfold(n, splits, shuffle=True):
            clf = Pipeline([('CSP', csp), ('LDA', lda)])
            #clf.set_params(CSP__reg=0.5)
            clf.fit(x[sub][tr], y[sub][tr])
            a = clf.score(x[sub][val], y[sub][val])
            a2 = clf.score(xt[sub], yt[sub])
            acc += a
            acc2 += a2

    print("{} got {}/{}".format(name, acc / (splits * subs), acc2 / (splits * subs)))


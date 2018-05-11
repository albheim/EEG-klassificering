"""
===========================================================================
Motor imagery decoding from EEG data using the Common Spatial Pattern (CSP)
===========================================================================
Decoding of motor imagery applied to EEG data decomposed using CSP.
Here the classifier is applied to features extracted on CSP filtered signals.
See http://en.wikipedia.org/wiki/Common_spatial_pattern and [1]_. The EEGBCI
dataset is documented in [2]_. The data set is available at PhysioNet [3]_.
References
----------
.. [1] Zoltan J. Koles. The quantitative extraction and topographic mapping
       of the abnormal components in the clinical EEG. Electroencephalography
       and Clinical Neurophysiology, 79(6):440--447, December 1991.
.. [2] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N.,
       Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer Interface
       (BCI) System. IEEE TBME 51(6):1034-1043.
.. [3] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
       Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000) PhysioBank,
       PhysioToolkit, and PhysioNet: Components of a New Research Resource for
       Complex Physiologic Signals. Circulation 101(23):e215-e220.
"""
# Authors: Martin Billinger <martin.billinger@tugraz.at>
#
# License: BSD (3-clause)

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

print(__doc__)

# #############################################################################
# # Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = -1.5, 2.5
event_id = dict(FA=0, LM=1, OB=2)

acc = 0

bin_size = 40
n_bins = 6
p_size = bin_size * n_bins

start = 700
end = 1600
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
qda = QuadraticDiscriminantAnalysis()
mlp = MLPClassifier(activation='tanh', solver='adam', alpha=1e-5,
                    learning_rate='invscaling', verbose=False, max_iter=500,
                    hidden_layer_sizes=(20, 10, 3), random_state=1)
svc = SVC(kernel='linear')
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

for sub in []:#[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]:
    x, y = data.load_single_sub(sub, cut=False)
    #x = x[:, :, 768:1280]
    y = np.where(y==1)[1]
    xt, yt = data.load_single_sub(sub, cut=False, study=False)
    yt = np.where(yt==1)[1]
    #print(x.shape, x_t.shape, y.shape)

    for i in range(steps):
        print("start")
        clf = Pipeline([('CSP', csp), ('LDA', lda)])
        #clf.set_params(CSP__reg=0.5)
        clf.fit(x[:, :, timepoints[i]:timepoints[i]+p_size], y)
        for j in range(steps):
            # Use scikit-learn Pipeline with cross_val_score function
            scores = clf.score(xt[:, :, timepoints[j]:timepoints[j]+p_size], yt)
            print("*", scores)
            heatmap[i,j] += np.mean(scores)

heatmap /= 18
print(heatmap)
np.savetxt("timepoints_sub5_18reps.csv", heatmap, delimiter=',')

csp = CSP(n_components=1, reg=None, log=True, norm_trace=False)
x, y = data.load_all(cut=False)
y = np.where(y==1)[1]
print(x.shape)
for i in timepoints:
    csp.fit_transform(x[:, :, timepoints[i]:timepoints[i]+p_size], y)

    layout = read_layout('EEG1005')
    csp.plot_patterns(info, layout=layout, ch_type='eeg',
                      units='Patterns (AU)', size=1.5)


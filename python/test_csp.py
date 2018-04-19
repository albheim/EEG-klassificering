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
subject = 5

x, y = data.load_single_sub(subject, cut=False, shuffle=False)
x_t = x[:, :, 768:1280]
y = np.where(y==1)[1]
print(x.shape, x_t.shape, y.shape)

info = create_info(
    ch_names=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FCz',
              'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2',
              'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'O2', 'PO10', 'Iz'],
    ch_types=['eeg' for _ in range(31)],
    sfreq=512
)

###############################################################################
# Classification with linear discrimant analysis

# Define a monte-carlo cross-validation generator (reduce variance):
scores = []
epochs_data = x
epochs_data_train = x_t
labels = y
print("**************epochs_data_train", epochs_data_train.shape)
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)

# Assemble a classifier
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

# Printing the results
print(labels)
nc = len(np.unique(labels))
N = len(labels)
pc = 1.0 / nc
qc = 1 - pc
class_balance = pc + 1.645 * (pc * qc / N)**0.5
print("Classification accuracy: %f / Chance level for equally "\
      "probable classes with p=0.05: %f" % (np.mean(scores),
                                           class_balance))

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)

layout = read_layout('EEG1005')
csp.plot_patterns(info, layout=layout, ch_type='eeg',
                  units='Patterns (AU)', size=1.5)

###############################################################################
# Look at performance over time

sfreq = info['sfreq']
w_length = int(sfreq * 0.5)   # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

scores_windows = []

for train_idx, test_idx in cv_split:
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
    X_test = csp.transform(epochs_data_train[test_idx])

    # fit classifier
    lda.fit(X_train, y_train)

    # running classifier: test classifier on sliding window
    score_this_window = []
    for n in w_start:
        X_test = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
        score_this_window.append(lda.score(X_test, y_test))
    scores_windows.append(score_this_window)

# Plot scores over time
w_times = (w_start + w_length / 2.) / sfreq + tmin

plt.figure()
plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
plt.axvline(0, linestyle='--', color='k', label='Onset')
plt.axhline(class_balance, linestyle='-', color='k', label='Chance')
plt.xlabel('time (s)')
plt.ylabel('classification accuracy')
plt.title('Classification score over time')
plt.legend(loc='lower right')
plt.show()

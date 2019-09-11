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
# lda = LinearDiscriminantAnalysis()
# qda = QuadraticDiscriminantAnalysis()
# mlp = MLPClassifier(activation='tanh', solver='adam', alpha=1e-5,
#                     learning_rate='invscaling', verbose=False, max_iter=500,
#                     hidden_layer_sizes=(20, 10, 3), random_state=1)
# svc = SVC(kernel='linear')
# csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# for sub in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]:
#     x, y = data.load_single_sub(sub, cut=False)
#     #x = x[:, :, 768:1280]
#     y = np.where(y==1)[1]
#     xt, yt = data.load_single_sub(sub, cut=False, study=False)
#     yt = np.where(yt==1)[1]
#     #print(x.shape, x_t.shape, y.shape)
#
#     for i in range(steps):
#         print("start")
#         clf = Pipeline([('CSP', csp), ('LDA', lda)])
#         #clf.set_params(CSP__reg=0.5)
#         clf.fit(x[:, :, timepoints[i]:timepoints[i]+p_size], y)
#         for j in range(steps):
#             # Use scikit-learn Pipeline with cross_val_score function
#             scores = clf.score(xt[:, :, timepoints[j]:timepoints[j]+p_size], yt)
#             print("*", scores)
#             heatmap[i,j] += np.mean(scores)
#
# heatmap /= 18
# print(heatmap)
# np.savetxt("timepoints_sub5_18reps.csv", heatmap, delimiter=',')

n_comp = 4
csp = CSP(n_components=n_comp, reg=None, log=True, norm_trace=False)
#x, y = data.load_all(cut=[start, end])
#y = np.where(y==1)[1]
#print(x.shape)

print(timepoints)

x, y = data.load_single([5], cut=False)
x = x[0]
y = np.where(y[0]==1)[1]
for i in range(steps):
    csp.fit_transform(x[:, :, timepoints[i]:timepoints[i] + p_size], y)

    for j in range(n_comp):
        layout = read_layout('EEG1005')
        csp.plot_patterns(info, colorbar=False, layout=layout, ch_type='eeg',
                        size=1.5, show=False, show_names=False, components=j,
                        title="{}".format(int((timepoints[i] - 768) / 0.512)))
        #plt.savefig('fig/sub9_time_{}_comp_{}.png'.format(i, j))




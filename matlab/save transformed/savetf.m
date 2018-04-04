clear; clc; close all
addpath('..')
addpath('../borrowed code')
[X,Y,n] = aux_load('Visual','06');

param.L = 8; param.Fs = 512; param.NFFT = 4096; param.NSTEP = 4;

X = aux_extr(X, 769:1024);
X = aux_transform(X, 'spec', param);
for i = 1:length(X)
    X{i} = permute(X{i}, [2 3 1]);
end

save('/lunarc/nobackup/users/albheim/EEG-klassificering/DATA/Modified/spectrogram/???', X)
save('/lunarc/nobackup/users/albheim/EEG-klassificering/DATA/Modified/spectrogram/???', Y)

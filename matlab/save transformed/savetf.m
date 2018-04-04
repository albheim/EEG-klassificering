clear; clc; close all
addpath('..')
addpath('../borrowed code')
[X,Y,n] = aux_load('Verbal','06');

param.L = 8; param.Fs = 512; param.NFFT = 4096; param.NSTEP = 32;

X = aux_extr(X, 769:1024);
X = aux_svd(X, 1);
X = aux_transform(X, 'spec', param);
for i = 1:length(X)
    X{i} = permute(X{i}, [2 3 1]);
end
X = aux_svd(X, 1);

X = aux_prep(X);
acc = aux_eval(X, Y);

clear; clc; close all
addpath('..')
addpath('../borrowed code')
addpath('../borrowed code/f_PlotEEG_BrainNetwork/f_PlotEEG_BrainNetwork')
[X,Y,n] = aux_load('Visual','05');

%param.L = 8; param.Fs = 512; param.NFFT = 4096; param.NSTEP = 4;

%X = aux_chan(X,[5 24]);
X = aux_extr(X, 769:1024);
%X = aux_deci(X,2);
%X = aux_transform(X, 'spec', param);
X = aux_covm(X);

for i = 1:10
    aij = X{i};

    nch = 31; %take care of this variable (nch must be according to matrix size you want to plot it)
    p = 0.03;   %proportion of weigthed links to keep for.
    aij = threshold_proportional(aij, p); %thresholding networks due to proportion p
    ijw = adj2edgeL(triu(aij));             %passing from matrix form to edge list form

    figure
    f_PlotEEG_BrainNetwork(nch, ijw, 'w_wn2wx');
end
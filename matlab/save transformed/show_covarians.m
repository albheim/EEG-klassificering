clear; clc; close all
addpath('..')
addpath('../borrowed code')
addpath('../albins')
addpath('../borrowed code/f_PlotEEG_BrainNetwork/f_PlotEEG_BrainNetwork')
[X,Y,n] = aux_load('Visual','05');

%param.L = 8; param.Fs = 512; param.NFFT = 4096; param.NSTEP = 4;

%X = aux_chan(X,[5 24]);
X = aux_extr(X, 769:1024);
%X = aux_deci(X,2);
%X = aux_transform(X, 'spec', param);
X = aux_covm(X);


figure
f_PlotEEG_BrainNetwork(31)

idx = [1, 12, 23, 80, 71, 96, 160, 171, 155];
figure
for i = 1:length(idx)
    aij = X{idx(i)};

    p = 0.01;   %proportion of weigthed links to keep for.
    aij = threshold_proportional(aij, p); %thresholding networks due to proportion p
    ijw = adj2edgeL(triu(aij));             %passing from matrix form to edge list form
    %ijw = layout_conv(ijw)
    subplot(3, 3, i);
    f_PlotEEG_BrainNetwork(31, ijw, 'w_wn2wx');
end


Xavg = sum(cat(3, X{:}), 3) / length(X);
[xx, yy] = meshgrid(1:31, 1:31);

figure;
colormap(gca,parula)
pcolor(xx,yy,Xavg);
title('average covariance between channels for sub 5 visual')
xlabel('channel')
ylabel('channel')
hcb = colorbar
ylabel(hcb, 'covariance')
clear; close all; clc

addpath(genpath('../../../FMSN35 - Spektralanalys/Matlab code'))
[FA,LM,OB] = load_data('Verbal','02');
label = FA.label;

NSTEP = 1; NFFT = 4096; Fs = 512; L = 64;
N = 2048/NSTEP + 1; Ntri = length(FA.trial);
% Mean value of data first, and then marginal
figure(1); clf
for i = 1:31
    subplot(4,8,i)
    data = zeros(1,2049);
    for j = 1:Ntri
        data = data + FA.trial{j}(i,:);
    end
    data = data/Ntri;
    [spec, TI, FI] = mtspectrogram(data,L,Fs,NFFT,NSTEP);
    S = sum(spec,2);
    plot(TI,S)
    axis tight;
    xlim([min(TI) max(TI)])
    title(label(i))
end
suptitle('Marginals of mean electrode signals, Faces');

figure(2); clf
for i = 1:31
    subplot(4,8,i)
    data = zeros(1,2049);
    for j = 1:Ntri
        data = data + LM.trial{j}(i,:);
    end
    data = data/Ntri;
    [spec, TI, FI] = mtspectrogram(data,L,Fs,NFFT,NSTEP);
    S = sum(spec,2);
    plot(TI,S)
    axis tight;
    xlim([min(TI) max(TI)])
    title(label(i))
end
suptitle('Marginals of mean electrode signals, Landmarks');

figure(3); clf
for i = 1:31
    subplot(4,8,i)
    data = zeros(1,2049);
    for j = 1:Ntri
        data = data + OB.trial{j}(i,:);
    end
    data = data/Ntri;
    [spec, TI, FI] = mtspectrogram(data,L,Fs,NFFT,NSTEP);
    S = sum(spec,2);
    plot(TI,S)
    axis tight;
    xlim([min(TI) max(TI)])
    title(label(i))
end
suptitle('Marginals of mean electrode signals, Objects');
%%
figure(4); clf
data = FA.trial{9}';
X = zeros(ceil(size(data,1)/NSTEP),size(data,2));
for i = 1:size(X,2)
    [S,~,~] = mtspectrogram(data(:,i),L,Fs,NFFT,NSTEP);
    X(:,i) = sum(S,2);
end
[U,sigma,V] = svd(X);
for i = 1:16
    subplot(4,4,i)
    plot(U(:,i))
    title(sprintf('PC %d',i))
end
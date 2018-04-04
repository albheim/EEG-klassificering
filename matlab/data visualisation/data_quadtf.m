clear; clc

addpath(genpath('../../../FMSN35 - Spektralanalys/Matlab code'))

[FA,LM,OB] = load_data('Verbal','09');

figure(1); clf
EEGd = LM;
nTri = length(EEGd.trial); nCh = length(EEGd.label);
N = length(EEGd.trial{1});

n = 4; NFFT = 4096/n; Fs = 512/n;

for i = 1:nCh
    data = zeros(N,1);
    for j = 1:nTri
        data = data + EEGd.trial{j}(i,:)';
    end
    ens_avg = hilbert(decimate(data/nTri,n));
    subplot(4,8,i)
    quadtf(ens_avg,'rih',1,Fs,NFFT);
    ylim([0 Fs/2]);
    title(EEGd.label{i},'interpreter','latex');
end
%%
figure(2); clf
for i = 1:nTri
    clf
    [nCh, N] = size(EEGd.trial{i});
    tt = (1:N)*4/N;
    for j = 1:nCh
        subplot(4,8,j)
        quadtf(decimate(EEGd.trial{i}(j,:)',n),'rihaczek',1,Fs,NFFT);
        ylim([0 Fs/2]);
        title(EEGd.label(j),'interpreter','latex')
    end
    h = suptitle(sprintf('Trial %d',i));
    set(h,'interpreter','latex')
    pause;
end
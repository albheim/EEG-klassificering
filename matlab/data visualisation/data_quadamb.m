clear; clc

addpath(genpath('../../../FMSN35 - Spektralanalys/Matlab code'))

[FA,LM,OB] = load_data('Verbal','09');

figure(1); clf
EEGd = OB;
nTri = length(EEGd.trial); nCh = length(EEGd.label);
N = length(EEGd.trial{1});

n = 8; NFFT = 4096/2/n; Fs = 512/n;

for i = 1:nCh
    data = zeros(N,1);
    for j = 1:nTri
        data = data + EEGd.trial{j}(i,:)';
    end
    ens_avg = hilbert(decimate(data/nTri,n));
    subplot(4,8,i)
    [A,TI,FI] = quadamb(ens_avg,'wigner',1,Fs,NFFT);
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
        quadamb(decimate(EEGd.trial{i}(j,:)',n),'wigner',1,Fs,NFFT);
        title(EEGd.label(j),'interpreter','latex')
    end
    h = suptitle(sprintf('Trial %d',i));
    set(h,'interpreter','latex')
    pause;
end
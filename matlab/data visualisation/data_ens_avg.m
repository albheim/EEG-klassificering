clear; clc

addpath(genpath('../../../FMSN35 - Spektralanalys/Matlab code'))

[FA,LM,OB] = load_data('Verbal','15');

figure(1); clf
EEGd = FA;
nTri = length(EEGd.trial); nCh = length(EEGd.label);
N = length(EEGd.trial{1});
for i = 1:nCh
    data = zeros(N,1);
    for j = 1:10
        data = data + EEGd.trial{j}(i,:)';
    end
    ens_avg = data/nTri;
    subplot(4,8,i)
    plot((1:N)*(4/N),ens_avg)
    title(EEGd.label{i},'interpreter','latex');
end

figure(2); clf
for i = 1:nTri
    clf
    [nCh, N] = size(EEGd.trial{i});
    tt = (1:N)*4/N;
    for j = 1:nCh
        subplot(4,8,j)
        plot(tt,EEGd.trial{i}(j,:)')
        title(EEGd.label(j),'interpreter','latex')
        ylim([-50 50])
    end
    h = suptitle(sprintf('Trial %d',i));
    set(h,'interpreter','latex')
    pause;
end
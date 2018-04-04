clear; clc; close all

addpath(genpath('../../../Matlab code'))

[FA, LM, OB] = load_data('Verbal','01');

%% Change channel each iteration
numTri = length(FA.trial); n = 8;
for i = 1:numTri
    [nCh, N] = size(FA.trial{i});
    for j = 1:nCh
        data = decimate(FA.trial{i}(j,:)',n);
        analyzets(data-mean(data),100,0.05,1,0);
        title(sprintf('Trial %d, channel %d',i,j));
        pause(0.3);
    end
end

%% Change trial each iteration

numTri = length(FA.trial);
for j = 1:31
    for i = 1:numTri
        data = decimate(FA.trial{i}(j,:)',n);
        analyzets(data-mean(data),20,0.05,2,1);
        title(['Trial ' num2str(i) ', Channel ' num2str(j) ' (' FA.label{j} ')']);
        pause(0.1);
    end
end
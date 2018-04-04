clear; clc

addpath(genpath('../../../FMSN35 - Spektralanalys/Matlab code'))
addpath(genpath('..'))
[FA,LM,OB] = load_data('Visual','18');
% Good - visual 05, visual 18, verbal 19
% Bad  - visual 14, verbal 16, verbal 20

%% Prepare data sets

% Choose channels
% Fz - 5; Cz - 15; Pz - 24
ch = 15; r = 8; nCh = length(ch); n_feat = 32;
nFA = length(FA.trial); nLM = length(LM.trial); nOB = length(OB.trial);
N = nFA + nLM + nOB;
XFA = zeros(nFA,n_feat*nCh);
XLM = zeros(nLM,n_feat*nCh);
XOB = zeros(nOB,n_feat*nCh);

for i = 1:nFA
    sampleInd = 769:1024;
    x = FA.trial{i}(:,sampleInd);
    for j = 1:nCh
        W = wigner1(hilbert(x(ch(j),:)'), 512, 2*length(sampleInd));
        Wtime = downsample(downsample(sum(W),4),8); 
        Wfreq = downsample(downsample(sum(W,2)',4),8);
        XFA(i,n_feat*(j-1)+(1:n_feat)) = [Wtime, Wfreq];
    end
end
for i = 1:nLM
    sampleInd = 769:1024;
    x = LM.trial{i}(:,sampleInd);
    for j = 1:nCh
        W = wigner1(hilbert(x(ch(j),:)'), 512, 2*length(sampleInd));
        Wtime = downsample(downsample(sum(W),4),8); 
        Wfreq = downsample(downsample(sum(W,2)',4),8);
        XLM(i,n_feat*(j-1)+(1:n_feat)) = [Wtime, Wfreq];
    end
end
for i = 1:nOB
    sampleInd = 769:1024;
    x = OB.trial{i}(:,sampleInd);
    for j = 1:nCh
        W = wigner1(hilbert(x(ch(j),:)'), 512, 2*length(sampleInd));
        Wtime = downsample(downsample(sum(W),4),8); 
        Wfreq = downsample(downsample(sum(W,2)',4),8);
        XOB(i,n_feat*(j-1)+(1:n_feat)) = [Wtime, Wfreq];
    end
end
clear x
% Apply transforms/feature extraction here

X = [XFA; XLM; XOB];
Y = [ones(size(XFA,1),1); 2*ones(size(XLM,1),1); 3*ones(size(XOB,1),1)];
t = templateSVM('Standardize',1);
ECOCModel = fitcecoc(X,Y,'Learners',t);
CVModel = crossval(ECOCModel);
[label,score] = kfoldPredict(CVModel);
disp('SVM');
sum(label == Y)/length(Y)
plotconfusion(ind2vec(Y'), ind2vec(label'))
title('Confusion matrix, SVM');
%{
% logistic
num_folds = 10;
indices = crossvalind('Kfold',N,num_folds);
label = zeros(N,1);
for i = 1:num_folds
    test = (indices == i); train = ~test;
    [b,dev,stats] = mnrfit(X(train,:),Y(train));
    label(i) = mnrval(b,X(test,:));
end
disp('Logistic');
sum(label==Y)/length(Y)
plotconfusion(ind2vec(Y'), ind2vec(label'))
title('Confusion matrix, logistic');
%}
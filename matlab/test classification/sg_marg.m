clear; clc

addpath(genpath('../../../FMSN35 - Spektralanalys/Matlab code'))
addpath(genpath('..'))
[FA,LM,OB] = load_data('Visual','05');
% Good - visual 05, visual 18, verbal 19
% Bad  - visual 14, verbal 16, verbal 20

%% Prepare data sets

% Choose channels
% Fz - 5; Cz - 15; Pz - 24
ch = 5; r = 8; nCh = length(ch); n_feat = 32;
nFA = length(FA.trial); nLM = length(LM.trial); nOB = length(OB.trial);
N = nFA + nLM + nOB;
XFA = zeros(nFA,n_feat*nCh);
XLM = zeros(nLM,n_feat*nCh);
XOB = zeros(nOB,n_feat*nCh);

for i = 1:nFA
    sampleInd = 769:1024;
    x = FA.trial{i}(:,sampleInd);
    for j = 1:nCh
        W = mtspectrogram(x(ch(j),:)', 8, 512, 2*length(sampleInd),4);
        Wtime = downsample(downsample(sum(W),4),4); 
        Wfreq = downsample(sum(W,2)',4);
        XFA(i,n_feat*(j-1)+(1:n_feat)) = [Wtime, Wfreq];
    end
end
for i = 1:nLM
    sampleInd = 769:1024;
    x = LM.trial{i}(:,sampleInd);
    for j = 1:nCh
        W = mtspectrogram(x(ch(j),:)', 8, 512, 2*length(sampleInd),4);
        Wtime = downsample(downsample(sum(W),4),4); 
        Wfreq = downsample(sum(W,2)',4);
        XLM(i,n_feat*(j-1)+(1:n_feat)) = [Wtime, Wfreq];
    end
end
for i = 1:nOB
    sampleInd = 769:1024;
    x = OB.trial{i}(:,sampleInd);
    for j = 1:nCh
        W = mtspectrogram(x(ch(j),:)', 8, 512, 2*length(sampleInd),4);
        Wtime = downsample(downsample(sum(W),4),4); 
        Wfreq = downsample(sum(W,2)',4);
        XOB(i,n_feat*(j-1)+(1:n_feat)) = [Wtime, Wfreq];
    end
end
clear x
% Apply transforms/feature extraction here

X = [XFA; XLM; XOB];
Y = [ones(size(XFA,1),1); 2*ones(size(XLM,1),1); 3*ones(size(XOB,1),1)];
%% ECOC/SVM -- linear
t = templateSVM('Standardize',1,'KernelFunction','linear');
ECOCModel = fitcecoc(X,Y,'Learners',t);
CVModel = crossval(ECOCModel);
[label,~] = kfoldPredict(CVModel);
fprintf('SVM acc.:\n\t %1.4f\n',sum(label == Y)/length(Y));
figure(1); clf
plotconfusion(ind2vec(Y'), ind2vec(label'))
title('Confusion matrix, SVM');
%% ECOC/SVM -- polynomial
t = templateSVM('Standardize',1,'KernelFunction','polynomial');
ECOCModel = fitcecoc(X,Y,'Learners',t);
CVModel = crossval(ECOCModel);
[label,~] = kfoldPredict(CVModel);
fprintf('SVM poly acc.:\n\t %1.4f\n',sum(label == Y)/length(Y));
figure(1); clf
plotconfusion(ind2vec(Y'), ind2vec(label'))
title('Confusion matrix, SVM/poly');
%% Lin. discriminant
DiscrModel = fitcdiscr(X,Y,'CrossVal','on');
[label,~] = kfoldPredict(DiscrModel);
fprintf('Discriminant acc.:\n\t %1.4f\n',sum(label == Y)/length(Y));
figure(2); clf
plotconfusion(ind2vec(Y'), ind2vec(label'))
title('Confusion matrix, Discr.');
%% Class. tree
TreeModel = fitctree(X,Y,'CrossVal','on');
[label,score] = kfoldPredict(TreeModel);
fprintf('Tree acc.:\n\t %1.4f\n',sum(label == Y)/length(Y));
figure(2); clf
plotconfusion(ind2vec(Y'), ind2vec(label'))
title('Confusion matrix, Tree');

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
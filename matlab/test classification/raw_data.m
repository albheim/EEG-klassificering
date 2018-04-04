clear; clc

addpath(genpath('../../../FMSN35 - Spektralanalys/Matlab code'))
[FA,LM,OB] = load_data('Visual','05');
% Good - visual 05, visual 18, verbal 19
% Bad  - visual 14, verbal 16, verbal 20

%% Prepare data sets

% Choose channels and decimate
% Fz - 5; Cz - 15; Pz - 24
ch = [5 24]; r = 8;

for i = 1:length(FA.trial)
    FA.trial{i} = FA.trial{i}(:,769:1024);
end
for i = 1:length(LM.trial)
    LM.trial{i} = LM.trial{i}(:,769:1024);
end
for i = 1:length(OB.trial)
    OB.trial{i} = OB.trial{i}(:,769:1024);
end

XFA = prepdata(decimate_eeg(FA,r),ch); 
XLM = prepdata(decimate_eeg(LM,r),ch); 
XOB = prepdata(decimate_eeg(OB,r),ch);

% Apply transforms/feature extraction here


% Split into sets
tR = 0.9; vR = 0.1;

[FAtrain,FAval] = splitdata(XFA,tR,vR);
[LMtrain,LMval] = splitdata(XLM,tR,vR);
[OBtrain,OBval] = splitdata(XOB,tR,vR);

Xtrain = [FAtrain; LMtrain; OBtrain];
Ytrain = [ones(size(FAtrain,1),1); ...
          2*ones(size(LMtrain,1),1); ...
          3*ones(size(OBtrain,1),1)];
X = [XFA; XLM; XOB];
%{
Y = zeros(size(X,1),3);
Y(1:size(XFA,1),1) = 1;
Y(size(XFA,1)+(1:size(XLM,1)),2) = 1;
Y(size(XFA,1)+size(XLM,1)+(1:size(XOB,1)),3) = 1;
%}
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
figure(2); clf
plotconfusion(ind2vec(Y'), ind2vec(label'))
title('Confusion matrix, SVM/poly');
%% Lin. discriminant
DiscrModel = fitcdiscr(X,Y,'CrossVal','on');
[label,~] = kfoldPredict(DiscrModel);
fprintf('Discriminant acc.:\n\t %1.4f\n',sum(label == Y)/length(Y));
figure(3); clf
plotconfusion(ind2vec(Y'), ind2vec(label'))
title('Confusion matrix, Discr.');
%% Class. tree
TreeModel = fitctree(X,Y,'CrossVal','on');
[label,score] = kfoldPredict(TreeModel);
fprintf('Tree acc.:\n\t %1.4f\n',sum(label == Y)/length(Y));
figure(4); clf
plotconfusion(ind2vec(Y'), ind2vec(label'))
title('Confusion matrix, Tree');

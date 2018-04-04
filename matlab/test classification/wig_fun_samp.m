clear; clc

addpath(genpath('../../../FMSN35 - Spektralanalys/Matlab code'))
addpath(genpath('..'))
[FA,LM,OB] = load_data('Visual','18');
% Good - visual 05, visual 18, verbal 19
% Bad  - visual 14, verbal 16, verbal 20

%% Prepare data sets

% Choose channels
% Fz - 5; Cz - 15; Pz - 24
ch = 15; r = 8; nCh = length(ch); 
nFA = length(FA.trial); nLM = length(LM.trial); nOB = length(OB.trial);
XFA = zeros(nFA,32*nCh);
XLM = zeros(nLM,32*nCh);
XOB = zeros(nOB,32*nCh);

for i = 1:nFA
    sampleInd = 769:1024;
    x = FA.trial{i}(:,sampleInd);
    for j = 1:nCh
        W = wigner1(hilbert(x(ch(j),:)'), 512, 2*length(sampleInd));
        W = downsample(downsample(W,64)',32)';
        XFA(i,(numel(W)*(j-1))+(1:numel(W))) = real(W(:))';
    end
end
for i = 1:nLM
    sampleInd = 769:1024;
    x = LM.trial{i}(:,sampleInd);
    for j = 1:nCh
        W = wigner1(hilbert(x(ch(j),:)'), 512, 2*length(sampleInd));
        W = downsample(downsample(W,64)',32)';
        XLM(i,(numel(W)*(j-1))+(1:numel(W))) = real(W(:))';
    end
end
for i = 1:nOB
    sampleInd = 769:1024;
    x = OB.trial{i}(:,sampleInd);
    for j = 1:nCh
        W = wigner1(hilbert(x(ch(j),:)'), 512, 2*length(sampleInd));
        W = downsample(downsample(W,64)',32)';
        XOB(i,(numel(W)*(j-1))+(1:numel(W))) = real(W(:))';
    end
end

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
t = templateSVM('Standardize',1);
ECOCModel = fitcecoc(X,Y,'Learners',t);
CVModel = crossval(ECOCModel);
[label,score] = kfoldPredict(CVModel);
sum(label == Y)/length(Y)
%{
%% training
[B,dev,stats] = mnrfit(Xtrain,Ytrain);

%% validation

Xval = [FAval; LMval; OBval];
Yval = [ones(size(FAval,1),1); ...
        2*ones(size(LMval,1),1); ...
        3*ones(size(OBval,1),1)];

Ypred = mnrval(B,Xval);
Ypred = mod(find(Ypred == max(Ypred,[],2)),3);
Ypred = Ypred + 3*(Ypred==0);
acc = sum((Ypred == Yval))/length(Yval); disp(acc)
%}

plotconfusion(ind2vec(Y'), ind2vec(label'))
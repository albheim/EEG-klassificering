function [ acc, predAcc ] = aux_eval( X,Y,plotflag,Xtest,Ytest )
%AUX_EVAL Evaluate features with 4 different ML algs
%   Get evaluation from linear SVM, poly SVM, LDA and tree

if nargin<3
    plotflag = 0;
end
acc = zeros(5,1);
test = (nargin > 3);
if test
    predAcc = zeros(5,1);
end

%% ECOC/SVM -- linear
t = templateSVM('Standardize',1,'KernelFunction','linear');
ECOCModel = fitcecoc(X,Y,'Learners',t);
CVModel = crossval(ECOCModel);
[label,~] = kfoldPredict(CVModel);
acc(1) = sum(label==Y)/length(Y);
if plotflag
    fprintf('SVM acc.:\n\t %1.4f\n',acc(1));
    h = figure(1); plotconfusion(ind2vec(Y'), ind2vec(label'))
    set(h,'Units','inches','Position',[0 6.6 4 4]);
    title('Confusion matrix, SVM');
end

if test
    Mdl = fitcecoc(X,Y,'Learners',t);
    label = predict(Mdl,Xtest);
    predAcc(1) = sum(label==Ytest)/length(Ytest);
end

%% Lin. discriminant
DiscrModel = fitcdiscr(X,Y,'CrossVal','on');
[label,~] = kfoldPredict(DiscrModel);
acc(2) = sum(label==Y)/length(Y);
if plotflag
    fprintf('Discriminant acc.:\n\t %1.4f\n',acc(2));
    h = figure(3); plotconfusion(ind2vec(Y'), ind2vec(label'))
    set(h,'Units','inches','Position',[0 2 4 4]);
    title('Confusion matrix, Discr.');
end

if test
    Mdl = fitcdiscr(X,Y);
    label = predict(Mdl,Xtest);
    predAcc(2) = sum(label==Ytest)/length(Ytest);
end
%% Class. tree
TreeModel = fitctree(X,Y,'CrossVal','on');
[label,~] = kfoldPredict(TreeModel);
acc(3) = sum(label==Y)/length(Y);
if plotflag
    fprintf('Tree acc.:\n\t %1.4f\n',acc(3));
    h = figure(4); plotconfusion(ind2vec(Y'), ind2vec(label'))
    set(h,'Units','inches','Position',[4 2 4 4]);
    title('Confusion matrix, Tree');
end

if test
    Mdl = fitctree(X,Y);
    label = predict(Mdl,Xtest);
    predAcc(3) = sum(label==Ytest)/length(Ytest);
end

%% Random forest, bagged
t = templateTree();
BagMdl = fitcensemble(X,Y,'Method','Bag','Learners',t,'CrossVal','on');
[label,~] = kfoldPredict(BagMdl);
acc(4) = sum(label==Y)/length(Y);
if plotflag
    fprintf('Tree acc.:\n\t %1.4f\n',acc(3));
    h = figure(4); plotconfusion(ind2vec(Y'), ind2vec(label'))
    set(h,'Units','inches','Position',[4 2 4 4]);
    title('Confusion matrix, Tree');
end

if test
    Mdl = fitcensemble(X,Y,'Method','Bag','Learners',t);
    label = predict(Mdl,Xtest);
    predAcc(4) = sum(label==Ytest)/length(Ytest);
end

%% Random forest, boosted
t = templateTree();
BagMdl = fitcensemble(X,Y,'Method','AdaBoostM2','Learners',t,'CrossVal','on');
[label,~] = kfoldPredict(BagMdl);
acc(5) = sum(label==Y)/length(Y);
if plotflag
    fprintf('Tree acc.:\n\t %1.4f\n',acc(3));
    h = figure(4); plotconfusion(ind2vec(Y'), ind2vec(label'))
    set(h,'Units','inches','Position',[4 2 4 4]);
    title('Confusion matrix, Tree');
end

if test
    Mdl = fitcensemble(X,Y,'Method','AdaBoostM2','Learners',t);
    label = predict(Mdl,Xtest);
    predAcc(5) = sum(label==Ytest)/length(Ytest);
end

end

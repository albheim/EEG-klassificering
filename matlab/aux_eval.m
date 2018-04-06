function [ acc ] = aux_eval( X,Y,plotflag )
%AUX_EVAL Evaluate features with 4 different ML algs
%   Get evaluation from linear SVM, poly SVM, LDA and tree

if nargin<3
    plotflag = 0;
end
acc = zeros(3,1);

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
%{
%% ECOC/SVM -- polynomial
t = templateSVM('Standardize',1,'KernelFunction','polynomial');
ECOCModel = fitcecoc(X,Y,'Learners',t);
CVModel = crossval(ECOCModel);
[label,~] = kfoldPredict(CVModel);
acc(4) = sum(label==Y)/length(Y);
if plotflag
    fprintf('SVM poly acc.:\n\t %1.4f\n',acc(4));
    h = figure(2); plotconfusion(ind2vec(Y'), ind2vec(label'))
    set(h,'Units','inches','Position',[4 6.6 4 4]);
    title('Confusion matrix, SVM/poly');
end
%}

end


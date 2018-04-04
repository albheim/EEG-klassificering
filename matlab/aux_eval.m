function [ acc ] = aux_eval( X,Y )
%AUX_CLASSIFY Evaluate features with 4 different ML algs
%   Get evaluation from linear SVM, poly SVM, LDA and tree

acc = zeros(4,1);
figure(1); clf

%% ECOC/SVM -- linear
t = templateSVM('Standardize',1,'KernelFunction','linear');
ECOCModel = fitcecoc(X,Y,'Learners',t);
CVModel = crossval(ECOCModel);
[label,~] = kfoldPredict(CVModel);
acc(1) = sum(label==Y)/length(Y);
fprintf('SVM acc.:\n\t %1.4f\n',acc(1));
h = figure(1); plotconfusion(ind2vec(Y'), ind2vec(label'))
set(h,'Units','inches','Position',[0 6.6 4 4]);
title('Confusion matrix, SVM');
%{
%% ECOC/SVM -- polynomial
t = templateSVM('Standardize',1,'KernelFunction','polynomial');
ECOCModel = fitcecoc(X,Y,'Learners',t);
CVModel = crossval(ECOCModel);
[label,~] = kfoldPredict(CVModel);
acc(2) = sum(label==Y)/length(Y);
fprintf('SVM poly acc.:\n\t %1.4f\n',acc(2));
h = figure(2); plotconfusion(ind2vec(Y'), ind2vec(label'))
set(h,'Units','inches','Position',[4 6.6 4 4]);
title('Confusion matrix, SVM/poly');
%}
%% Lin. discriminant
DiscrModel = fitcdiscr(X,Y,'CrossVal','on');
[label,~] = kfoldPredict(DiscrModel);
acc(3) = sum(label==Y)/length(Y);
fprintf('Discriminant acc.:\n\t %1.4f\n',acc(3));
h = figure(3); plotconfusion(ind2vec(Y'), ind2vec(label'))
set(h,'Units','inches','Position',[0 2 4 4]);
title('Confusion matrix, Discr.');
%% Class. tree
TreeModel = fitctree(X,Y,'CrossVal','on');
[label,~] = kfoldPredict(TreeModel);
acc(4) = sum(label==Y)/length(Y);
fprintf('Tree acc.:\n\t %1.4f\n',acc(4));
h = figure(4); plotconfusion(ind2vec(Y'), ind2vec(label'))
set(h,'Units','inches','Position',[4 2 4 4]);
title('Confusion matrix, Tree');

end


clear; clc; close all
addpath('../')

visStr = {'01','02','03','04','05','06','07','08','09',...
    '11','12','13','14','15','16','17','18','19'};
verStr = {'01','02','06','07','08','09','10','11','12',...
    '14','15','16','17','18','19','20','21','22'};

acc = zeros(36,3);
predAcc = zeros(36,3);
for i = 1:36
    if i < 19
        type = 'Visual';
        num = visStr{i};

    else
        type = 'Verbal';
        num = verStr{i-18};
    end
    fprintf(['Subject ' num ', ' type '\n']);
    [X,Y,n] = aux_load(type,num);
    [Xtest,Ytest,~] = aux_load(type,num,1);
    N = sum(n); r = 16;

    X = aux_extr(X,769:1280);
    Xtest = aux_extr(Xtest,769:1280);
    %X = aux_chan(X,15);
    %X = aux_svd(X,1:2);
    %X = aux_deci(X,r);
    %Xtest = aux_deci(Xtest,r);
    %X = aux_feat(X);
    %X = aux_covm(X);
    %X = aux_chan(X,1:2);

    X = aux_prep(X);
    Xtest = aux_prep(Xtest);
    [a, pa] = aux_eval(X,Y,0,Xtest,Ytest);
    acc(i,:) = a'; predAcc(i,:) = pa';
    fprintf(['\tAcc. on val. data: ' num2str(acc(i,:)) ...
        '\n\tPrediction on test: ' num2str(predAcc(i,:)) '\n\n'])
end
disp(mean(acc))
fprintf(['\n\nMean accuracy on validation data: ' num2str(mean(acc)) ...
    '\nMean prediction accuracy on test: ' num2str(mean(predAcc)) '\n'])

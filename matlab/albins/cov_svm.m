clear; clc; close all

addpath('../');

visStr = {'01','02','03','04','05','06','07','08','09',...
    '11','12','13','14','15','16','17','18','19'};
   
type = 'Visual';

acc = zeros(18,2);
for i = 1:18
    num = visStr{i};
    [Xt,Yt,n] = aux_load('Visual',num);
    [X,Y,nt] = aux_load('Visual',num,1);
    r=16;
    X = aux_extr(X,769:1280);
    Xt = aux_extr(Xt,769:1280);
    %X = aux_chan(X,15);
    %X = aux_svd(X,1:2);
    X = aux_deci(X,r);
    Xt = aux_deci(Xt,r);
    %X = aux_feat(X);

    X = aux_prep(X);
    Xt = aux_prep(Xt);
    
    t = templateSVM('Standardize',1,'KernelFunction','linear');
    ECOCModel = fitcecoc(X,Y,'Learners',t);
    CVModel = crossval(ECOCModel);
    [label,~] = kfoldPredict(CVModel);
    acc(i,1) = sum(label==Y)/length(Y)';
    [label,~] = predict(ECOCModel, Xt);
    acc(i,2) = sum(label==Yt)/length(Yt)';
end
disp(mean(acc))

% Feature extraction thing is 45% with 2 and 3 channels at 8x decimation


clear; clc; close all

visStr = {'01','02','03','04','05','06','07','08','09',...
    '11','12','13','14','15','16','17','18','19'};
verStr = {'01','02','06','07','08','09','10','11','12',...
    '14','15','16','17','18','19','20','21','22'};

acc = zeros(36,3);
for i = 1:36
    if i < 19
        type = 'Visual';
        num = visStr{i};
    else
        type = 'Verbal';
        num = verStr{i-18};
    end
    [X,Y,n] = aux_load(type,num);
    N = sum(n); r = 8;

    X = aux_extr(X,769:1024);
    %X = aux_chan(X,15);
    X = aux_svd(X,1:2);
    X = aux_deci(X,r);
    %X = aux_feat(X);
    %X = aux_covm(X);
    %X = aux_chan(X,1:2);

    X = aux_prep(X);
    acc(i,:) = aux_eval(X,Y,0)';
end
disp(mean(acc))

% Feature extraction thing is 45% with 2 and 3 channels at 8x decimation
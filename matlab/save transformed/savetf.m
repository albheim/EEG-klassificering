clear; clc; close all
addpath('..')
addpath('../borrowed code')
sub = '05';
[X,Y,n] = aux_load('Visual',sub);

param.L = 8; param.Fs = 512; param.NFFT = 1024; param.NSTEP = 8; 
param.method = 'l-ind'; param.NW = 3;

%X = aux_chan(X,[5 24 29]);
X = aux_extr(X, 769:1536);
%X = aux_deci(X,2);
%X = aux_svd(X, 1);
for method = ["spec", "wig", "amb", "cwt", "slep"]
    Xt = aux_transform(X, method, param);

    for i = 1:length(Xt)
        Xt{i} = permute(Xt{i}, [2 3 1]);
    end
    method
    size(Xt{1})

    %save(sprintf('C:\\Users\\Albin Heimerson\\Desktop\\exjobb\\DATA\\Modified\\spectogram\\%s_%s.mat', method, sub), 'Xt', 'Y', '-v7.3')

    save(sprintf('/lunarc/nobackup/users/albheim/EEG-klassificering/DATA/Modified/spectogram/%s_%s.mat', method, sub), 'X', 'Y', '-v7.3')
end



clear; clc; close all
addpath('..')
addpath('../borrowed code')
[X,Y,n] = aux_load('Visual','05');

param.L = 8; param.Fs = 512; param.NFFT = 4096; param.NSTEP = 8;

%X = aux_chan(X,[5 24]);
X = aux_extr(X, 769:1024);
X = aux_deci(X,2);
X = aux_transform(X, 'spec', param);

for i = 1:length(X)
    X{i} = permute(X{i}, [2 3 1]);
end

size(X{1})

%save('C:\Users\Albin Heimerson\Desktop\exjobb\DATA\Modified\spectogram\X6.mat', 'X','-v7.3')
%save('C:\Users\Albin Heimerson\Desktop\exjobb\DATA\Modified\spectogram\Y6.mat', 'Y','-v7.3')

save('/lunarc/nobackup/users/albheim/EEG-klassificering/DATA/Modified/spectogram/X5.mat', 'X', '-v7.3')
save('/lunarc/nobackup/users/albheim/EEG-klassificering/DATA/Modified/spectogram/Y5.mat', 'Y', '-v7.3')

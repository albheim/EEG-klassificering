clear; clc; close all
addpath('..')
addpath('../borrowed code')

for idx = [1 2 3 4 5 6 7 8 9 11 12 13 14 15 16 17 18 19]
    sub = sprintf('%02d', idx);
    [X,Y,n] = aux_load('Visual',sub);

    param.L = 8; param.Fs = 512; param.NFFT = 512; param.NSTEP = 1; 
    param.method = 'l-ind'; param.NW = 3;

    %X = aux_chan(X,[5 24 29]);
    X = aux_extr(X, 769:1536);
    X = aux_deci(X,4);
    %X = aux_svd(X, 1);

    ms = ["spec", "wig", "amb", "cwt", "slep"]
    for method = ms
        Xt = aux_transform(X, method, param);

        for i = 1:length(Xt)
            Xt{i} = real(permute(Xt{i}, [2 3 1]));
        end
        method
        size(Xt{1})

        %save(sprintf('C:\\Users\\Albin Heimerson\\Desktop\\exjobb\\DATA\\Modified\\spectogram\\%s_%s.mat', method, sub), 'Xt', 'Y', '-v7.3')

        save(sprintf('/lunarc/nobackup/users/albheim/EEG-klassificering/DATA/Modified/spectogram/%s_%s.mat', method, sub), 'Xt', 'Y', '-v7.3')
    end
end


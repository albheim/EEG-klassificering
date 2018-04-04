clear; clc

addpath(genpath('../../../FMSN35 - Spektralanalys/Matlab code'))
addpath(genpath('~/Documents/EEG-data'))
addpath(genpath('C:\Users\damir\Documents\EEG-data'))

str = 'Verbal/Subj09_CleanData_study_';
load([str 'FA.mat']); load([str 'LM.mat']); load([str 'OB.mat']);
data = Subj09_CleanData_study_FA;

label = data.label;
data = data.trial{25};

figure(1); clf
for i = 1:31
    subplot(4,8,i)
    plot(data(i,:))
    title(label(i))
    axis tight;
end
%{
figure(2); clf
for i = 1:31
    subplot(4,8,i)
    quadtf(data(i,:), 'wigner', 1, 512, 4096);
    title(label(i))
    ylim([0 40])
end
%}

NSTEP = 8; NFFT = 4096; Fs = 512;
N = 2048/NSTEP + 1;
% start with faces
data_FA = Subj09_CleanData_study_FA;
figure(3); clf; figure(6); clf
FA_data = data_FA.trial{60*ceil(rand)};
for i = 1:31
    %FA_data = zeros(1,2049);
    %{
    for j = 1:1
        data = data_FA.trial{60*ceil(rand)};
        FA_data = FA_data + data(i,:);
    end
    %}
    [FA_spec, TI, FI] = mtspectrogram(FA_data(i,:),64,Fs,NFFT,NSTEP);
    figure(3)
    subplot(4,8,i)
    S = FA_spec;
    c=[min(min(S)) max(max(S))];
    pcolor(TI,FI,S') 
    shading interp
    caxis(c)
    title(label(i))
    ylim([0 35]);
    figure(6)
    subplot(4,8,i)
    plot(TI,sum(S,2))
    title(label(i))
end
%%
% then landmarks
data_LM = Subj09_CleanData_study_LM;
figure(4); clf
for i = 1:31
    LM_data = zeros(1,2049);
    subplot(4,8,i)
    for j = 1:1
        data = data_LM.trial{60*ceil(rand)};
        LM_data = LM_data + data(i,:);
    end
    LM_data = LM_data/60;
    [LM_spec, TI, FI] = mtspectrogram(LM_data,64,Fs,NFFT,NSTEP);
    S = LM_spec;
    c=[min(min(S)) max(max(S))];
    pcolor(TI,FI,S')   
    shading interp
    caxis(c)
    title(label(i))
    ylim([0 35]);
end
%%
% then objects
data_OB = Subj09_CleanData_study_OB;
figure(5); clf
for i = 1:31
    OB_data = zeros(1,2049);
    subplot(4,8,i)
    for j = 1:1
        data = data_OB.trial{60*ceil(rand)};
        OB_data = OB_data + data(i,:);
    end
    OB_data = OB_data/60;
    [OB_spec, TI, FI] = mtspectrogram(OB_data,64,Fs,NFFT,NSTEP);
    S = OB_spec;
    c=[min(min(S)) max(max(S))];
    pcolor(TI,FI,S') 
    shading interp
    caxis(c)
    title(label(i))
    ylim([0 35]);
end

%% 

featureVec


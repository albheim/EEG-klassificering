clear; clc

addpath(genpath('../../../FMSN35 - Spektralanalys/Matlab code'))

n = 8;
NSTEP = 2; NFFT = 4096/2/n; Fs = 512/n; L = 64/n; N = 1024/NSTEP/n;
data_ind = (NFFT/8+1:3*NFFT/8)';

delta_ind = ceil(0.5*2/Fs*N):floor(4*2/Fs*N);
theta_ind = ceil(4*2/Fs*N):floor(8*2/Fs*N);
alpha_ind = ceil(8*2/Fs*N):floor(12*2/Fs*N);
bSMR_ind = ceil(12*2/Fs*N):floor(15*2/Fs*N);
bmid_ind = ceil(15*2/Fs*N):floor(18*2/Fs*N);
bhigh_ind = ceil(22*2/Fs*N):floor(32*2/Fs*N);

for i = 1:22
    if i < 10
        str = ['0' num2str(i)];
    else
        str = num2str(i);
    end
    [FA,LM,OB] = load_data('Verbal',str);
    if ~isequal(FA,0)
        numCh = length(FA.label);
        numFA = length(FA.trial); FAcell = cell(numFA,1); 
        numLM = length(LM.trial); LMcell = cell(numLM,1); 
        numOB = length(OB.trial); OBcell = cell(numOB,1);
        for j = 1:numFA
            FAspec = zeros(numCh,NFFT/4,6);
            for k = 1:numCh
                dec_data = decimate(FA.trial{j}(k,:)',n);
                [spec,~,~] = mtspectrogram(dec_data,L,Fs,NFFT,NSTEP);
                FAspec(k,:,1) = mean(spec(delta_ind,data_ind));
                FAspec(k,:,2) = mean(spec(theta_ind,data_ind));
                FAspec(k,:,3) = mean(spec(alpha_ind,data_ind));
                FAspec(k,:,4) = mean(spec(bSMR_ind,data_ind));
                FAspec(k,:,5) = mean(spec(bmid_ind,data_ind));
                FAspec(k,:,6) = mean(spec(bhigh_ind,data_ind));
            end
            FAcell{j} = FAspec;
        end
        for j = 1:numLM
            LMspec = zeros(numCh,NFFT/4,6);
            for k = 1:numCh
                dec_data = decimate(LM.trial{j}(k,:)',n);
                [spec,~,~] = mtspectrogram(dec_data,L,Fs,NFFT,NSTEP);
                LMspec(k,:,1) = mean(spec(delta_ind,data_ind));
                LMspec(k,:,2) = mean(spec(theta_ind,data_ind));
                LMspec(k,:,3) = mean(spec(alpha_ind,data_ind));
                LMspec(k,:,4) = mean(spec(bSMR_ind,data_ind));
                LMspec(k,:,5) = mean(spec(bmid_ind,data_ind));
                LMspec(k,:,6) = mean(spec(bhigh_ind,data_ind));
            end
            LMcell{j} = LMspec;
        end
        for j = 1:numOB
            OBspec = zeros(numCh,NFFT/4,6);
            for k = 1:numCh
                dec_data = decimate(OB.trial{j}(k,:)',n);
                [spec,~,~] = mtspectrogram(dec_data,L,Fs,NFFT,NSTEP);
                OBspec(k,:,1) = mean(spec(delta_ind,data_ind));
                OBspec(k,:,2) = mean(spec(theta_ind,data_ind));
                OBspec(k,:,3) = mean(spec(alpha_ind,data_ind));
                OBspec(k,:,4) = mean(spec(bSMR_ind,data_ind));
                OBspec(k,:,5) = mean(spec(bmid_ind,data_ind));
                OBspec(k,:,6) = mean(spec(bhigh_ind,data_ind));
            end
            OBcell{j} = OBspec;
        end
        data.FA = FAcell;
        data.LM = LMcell;
        data.OB = OBcell;
        fileStr = ['shareddata/Subj' str '_Verbal_spec8'];
        save(fileStr,'data');
    end
    [FA,LM,OB] = load_data('Visual',str);
    if ~isequal(FA,0)
        numCh = length(FA.label);
        numFA = length(FA.trial); FAcell = cell(numFA,1); 
        numLM = length(LM.trial); LMcell = cell(numLM,1); 
        numOB = length(OB.trial); OBcell = cell(numOB,1);
        for j = 1:numFA
            FAspec = zeros(numCh,NFFT/4,6);
            for k = 1:numCh
                dec_data = decimate(FA.trial{j}(k,:)',n);
                [spec,~,~] = mtspectrogram(dec_data,L,Fs,NFFT,NSTEP);
                FAspec(k,:,1) = mean(spec(delta_ind,data_ind));
                FAspec(k,:,2) = mean(spec(theta_ind,data_ind));
                FAspec(k,:,3) = mean(spec(alpha_ind,data_ind));
                FAspec(k,:,4) = mean(spec(bSMR_ind,data_ind));
                FAspec(k,:,5) = mean(spec(bmid_ind,data_ind));
                FAspec(k,:,6) = mean(spec(bhigh_ind,data_ind));
            end
            FAcell{j} = FAspec;
        end
        for j = 1:numLM
            LMspec = zeros(numCh,NFFT/4,6);
            for k = 1:numCh
                dec_data = decimate(LM.trial{j}(k,:)',n);
                [spec,~,~] = mtspectrogram(dec_data,L,Fs,NFFT,NSTEP);
                LMspec(k,:,1) = mean(spec(delta_ind,data_ind));
                LMspec(k,:,2) = mean(spec(theta_ind,data_ind));
                LMspec(k,:,3) = mean(spec(alpha_ind,data_ind));
                LMspec(k,:,4) = mean(spec(bSMR_ind,data_ind));
                LMspec(k,:,5) = mean(spec(bmid_ind,data_ind));
                LMspec(k,:,6) = mean(spec(bhigh_ind,data_ind));
            end
            LMcell{j} = LMspec;
        end
        for j = 1:numOB
            OBspec = zeros(numCh,NFFT/4,6);
            for k = 1:numCh
                dec_data = decimate(OB.trial{j}(k,:)',n);
                [spec,~,~] = mtspectrogram(dec_data,L,Fs,NFFT,NSTEP);
                OBspec(k,:,1) = mean(spec(delta_ind,data_ind));
                OBspec(k,:,2) = mean(spec(theta_ind,data_ind));
                OBspec(k,:,3) = mean(spec(alpha_ind,data_ind));
                OBspec(k,:,4) = mean(spec(bSMR_ind,data_ind));
                OBspec(k,:,5) = mean(spec(bmid_ind,data_ind));
                OBspec(k,:,6) = mean(spec(bhigh_ind,data_ind));
            end
            OBcell{j} = OBspec;
        end
        data.FA = FAcell;
        data.LM = LMcell;
        data.OB = OBcell;
        fileStr = ['shareddata/Subj' str '_Visual_spec8'];
        save(fileStr,'data');
    end
end

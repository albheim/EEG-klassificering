function [ X,Y,n ] = aux_load( dType, num, testflag )
%LOAD_DATA Load EEG data from a specified subject

% dType: 	Either Verbal or Visual, must be a string
% num:  	Subject number, typically 01-22, also string 
% testflag:	1 if you want the test data set (optional, default 0)
%
% X:     	Predictor matrix (size NxP)
% Y:     	Class vector from 1 to K (K = 3, size Nx1)
% n:     	3x1 vector containing nFA, nLM, nOB (N = sum(n))

addpath(genpath('~/Documents/EEG-data'))
addpath(genpath('C:\Users\damir\Documents\EEG-data'))
addpath(genpath('C:\Users\Albin Heimerson\Desktop\exjobb\DATA'))
addpath(genpath('/lunarc/nobackup/users/albheim/EEG-klassificering/DATA'))

if nargin < 3
    testflag = 0
end

if (~ischar(dType))
    error('The input argument type must be a character array!');
end

if testflag
	str = [dType '/Subj' num '_CleanData_test_'];
else
	str = [dType '/Subj' num '_CleanData_study_'];
end

try
    FA = load([str 'FA.mat']); vars = fields(FA); data_FA = FA.(vars{1}).trial';
    LM = load([str 'LM.mat']); vars = fields(LM); data_LM = LM.(vars{1}).trial';
    OB = load([str 'OB.mat']); vars = fields(OB); data_OB = OB.(vars{1}).trial';
    n = [length(data_FA); length(data_LM); length(data_OB)];
    Y = [ones(n(1),1); 2*ones(n(2),1); 3*ones(n(3),1)];
    X = [data_FA; data_LM; data_OB];
catch
    error(['Combination of type and subject didn''t exist. ',...
             'Setting output to 0.'])
end

end


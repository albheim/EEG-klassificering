function [ data_FA, data_LM, data_OB ] = load_data( type, num )
%LOAD_DATA Load EEG data from a specified subject

% type:     Either Verbal or Visual, must be a string
% num:      Subject number, typically 01-22, also string 
%
% data_FA:  Faces data
% data_LM:  Landmarks data
% data_OB:  Objects data

addpath(genpath('~/Documents/EEG-data'))
addpath(genpath('C:\Users\damir\Documents\EEG-data'))

if (~ischar(type))
    error('The input argument type must be a character array!');
end

str = [type '/Subj' num '_CleanData_study_'];

try
    FA = load([str 'FA.mat']); vars = fields(FA); data_FA = FA.(vars{1});
    LM = load([str 'LM.mat']); vars = fields(LM); data_LM = LM.(vars{1});
    OB = load([str 'OB.mat']); vars = fields(OB); data_OB = OB.(vars{1});
catch
    warning(['Combination of type and subject didn''t exist. ',...
             'Setting data structs to 0.'])
    data_FA = [];
    data_LM = [];
    data_OB = [];
end

end


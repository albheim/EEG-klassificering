function [ dataOut ] = decimate_eeg( data, r )
%DECIMATE_EEG Decimates an EEG data set
%   Input and output EEG data cells

if isempty(data), return, end

dataOut = data;

for i = 1:length(data.trial)
    [n,m] = size(data.trial{i});
    tempTrial = zeros(n,ceil(m/r));
    for j = 1:n
        tempTrial(j,:) = decimate(data.trial{i}(j,:)',r)';
    end
    dataOut.trial{i} = tempTrial;
end

end
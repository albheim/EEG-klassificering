function out = aux_deci( in, r )
%DECIMATE_EEG Decimates an EEG data set
%   Input and output EEG data cells

if isempty(in), return, end

out = cell(size(in));

for i = 1:length(in)
    [n,m] = size(in{i});
    out{i} = zeros(n,ceil(m/r));
    for j = 1:n
        out{i}(j,:) = decimate(in{i}(j,:)',r)';
    end
end

end
function out = aux_svd( in, m )
%AUX_SVD Performs SVD on EEG data
%   Performs SVD cell (trial) wise and gives the m specified components

if isempty(in), return, end

out = cell(size(in));

for i = 1:length(in)
    [~,~,v] = svd(in{i});
    out{i} = v(:,m)';
end

end
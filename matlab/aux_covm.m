function out = aux_covm( in )
%AUX_COVM Takes data matrices and returns covariance matrix

if isempty(in), return, end

out = cell(size(in));

for i = 1:length(in)
    [n,~] = size(in{i});
    for j = 1:n
        out{i} = cov(in{i}');
    end
end

end
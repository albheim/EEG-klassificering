function out = aux_extr( in, sample_ind )
%AUX_EXTR Extract part of trial time series

out = cell(size(in));

for i = 1:length(in)
    out{i} = in{i}(:,sample_ind);
end

end


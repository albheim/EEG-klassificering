function out = aux_chan( in, ch )
%AUX_EXTR Extract part of trial time series

out = cell(size(in));

for i = 1:length(in)
    out{i} = in{i}(ch,:);
end

end


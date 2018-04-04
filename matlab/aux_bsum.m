function out = aux_bsum(A,block_nrows, block_ncols)
out = squeeze(sum(reshape(sum(reshape(A,block_nrows,[])),...
                    size(A,1)/block_nrows,block_ncols,[]),2));
end
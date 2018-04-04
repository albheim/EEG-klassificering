function [ out ] = aux_prep( in )
%PREPDATA Splits stupid cell data into reasonable input data

out = cell2mat(cellfun(@(x) reshape(x,1,numel(x)),in,...
    'UniformOutput',false));

end


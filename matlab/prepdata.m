function [ X ] = prepdata( data, ch )
%PREPDATA Splits stupid cell data into reasonable input data

dataExtr = @(x,ch) reshape(x(ch,:),1,length(ch)*size(x,2));
X = cell2mat(cellfun(@(x) dataExtr(x,ch),data.trial,'UniformOutput',false)');

end


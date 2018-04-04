function [ Xtrain, Xval, Xtest ] = splitdata( X, trainRatio, valRatio, testRatio )
%SPLITDATA Split data into sets
%   X is N by P matrix. Split along N

[N,~] = size(X);
if nargin > 3
    [trainInd, valInd, testInd] = dividerand(N, trainRatio, valRatio, testRatio);
    Xtest = X(testInd,:);
else
    [trainInd, valInd, ~] = dividerand(N, trainRatio, valRatio, 0);
end

Xtrain = X(trainInd,:);
Xval = X(valInd,:);

end


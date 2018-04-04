function out = aux_feat(in)
%AUX_FEAT Extract features to feature vector
% For now, only one channel

if isempty(in), return, end

out = cell(size(in));

for i = 1:length(in)
    [n,~] = size(in{i});
    out{i} = zeros(n,19);
    for j = 1:n
        out{i}(j,:) = feature_extract(in{i}(j,:)');
    end
end

end

function feat = feature_extract(x)
%% Time features
% feat 1: mean
feat(1) = mean(x);

% feat 2: variance
feat(end+1) = var(x);

% feat 3: skewness
feat(end+1) = skewness(x);

% feat 4: kurtosis
feat(end+1) = kurtosis(x);

% feat 5-7: AR coefficients
[a,e] = arburg(x,3);
feat(end+1:end+3) = a(2:4);

% feat 8: white noise variance
feat(end+1) = e;

%% Time-frequency features
nfft = 2^(2*nextpow2(length(x)));
W = quadtf(x,'wig',1,512,nfft);
%W = mtspectrogram(hilbert(x),4,512,nfft,1);

% feat 9-11: flux in time/freq/diag directions
feat(end+1) = mean(mean(abs(W-circshift(W,[1 0]))));
feat(end+1) = mean(mean(abs(W-circshift(W,[0 1]))));
feat(end+1) = mean(mean(abs(W-circshift(W,[1 1]))));

% feat 12: spectral flatness
feat(end+1) = geo_mean(abs(W(:)))/mean(W(:));

% feat 13: mean Minkowski distance, p = 2
feat(end+1) = mean(pdist(W,'minkowski',2))/nfft;

% feat 14: Normalized Renyi entropy, alpha = 3
feat(end+1) = renyi(W);

% feat 15: Shannon entropy
feat(end+1) = -mean(mean(W.*log2(abs(W))));

% feat 16-19: IF statistical features
[~,i] = max(W); f = (nfft-i)/(2*nfft);
feat(end+1) = mean(f);
feat(end+1) = var(f);
feat(end+1) = skewness(f);
feat(end+1) = kurtosis(f);

% feat 20-25: BW image analysis stuff

end
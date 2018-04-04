function [S,TI,FI]=mtspectrogram(X,WIN,Fs,NFFT,NSTEP,wei)

% MTSPECTROGRAM MTSpectrogram 
%    [S,TI,FI]=mtspectrogram(X,WIN,Fs,NFFT,NSTEP,wei) calculates the spectrogram using the 
%    short-time Fourier transform of the (NN X 1) data vector X and a
%    window WIN. A length of a normalized Hanning window can be specified
%    as WIN. Any other window function can be also set as the parameter
%    WIN, or alternatively a set of multitapers as a matrix with a chosen weighting vector, wei.  
%
%    S:    Output spectrogram, matrix of sixe (NN X NFFT/2).
%    TI:   Time vector (NN X 1).
%    FI:   Frequency vector (NFFT/2 X 1).
%    X:    Data sequence, must be of shorter length than NFFT. 
%    WIN:  A data window of length L less than NN. If the length is 
%          specified as the parameter WIN=L, a Hanning window of length L is used. A matrix
%          of size (L X K) specifiy a set of multitapers. 
%    Fs:   Sample frequency, default Fs=1. 
%    NFFT: The number of FFT-samples, default NFFT=1024.
%    NSTEP:Sample step to the next frame, default=1. 
%    wei:  Weights of the different multitaper spectrograms, size (K X 1), default wei(k)=1/K, k=1 ... K.

if nargin<2
    'Error: No data input and window'
end
[L,K]=size(WIN);
if L<K
  [L,K]=size(WIN');
  WIN=WIN';
end
if L==1 && K==1
    L=WIN;
    WIN=hanning(WIN);
    WIN=WIN./sqrt(WIN'*WIN);
end
if nargin<6
    wei=ones(K,1)/K;
end

if nargin<5
    NSTEP=1;
end

if nargin<4
    NFFT=1024;
end
if nargin<3
    Fs=1;
end

[NN,M]=size(X);
if M>NN
    X=transpose(X);
    NN=M;
end
wei=wei(:);

x=[zeros(fix(L/2),1);X;zeros(ceil(L/2),1)];
nfreq = ceil((length(x)-L)/NSTEP);

% Vectorised version. Works, but is slightly actually slower.
%{
ind = bsxfun(@plus,1:NSTEP:length(x)-L,(1:L)');
testdata = x(ind);
testdata = permute(repelem(testdata,1,1,K),[1 3 2]);
X1 = permute(fft(WIN.*testdata,NFFT),[1 3 2]);
F1 = real(X1).^2 + imag(X1).^2;
repwei = permute(repmat(wei,1,NFFT,nfreq),[2 3 1]);
Fsave = dot(F1,repwei,3)';
S = Fsave(:,1:NFFT/2)/Fs;
%}

j = 1; NFFTh = NFFT/2;
S=zeros(NFFTh,nfreq);
for i=1:NSTEP:length(x)-L
  testdata=repmat(x(i+1:i+L),1,K);
  X1=fft(WIN.*testdata,NFFT);
  F1 = real(X1).^2 + imag(X1).^2;
  Fsave=F1*wei;
  S(:,j)=Fsave(1:NFFTh);
  j = j+1;
end

% Correcting for sampling frequency
S=S'/Fs;

TI=(1:NSTEP:length(x)-L)'/Fs;
FI=(0:NFFTh-1)'/(NFFT)*Fs;
%{
c=[min(min(S)) max(max(S))];
pcolor(TI,FI,S')  
shading interp
caxis(c)
ylabel('Frequency (Hz)')
xlabel('Time (s)')
title('Spectrogram')
%}
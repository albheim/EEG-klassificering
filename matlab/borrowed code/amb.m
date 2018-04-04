function [ A, t, f ] = amb( x, fs, nfreq )
%AMB -- Compute the ambiguity function of input data
%
%   Usage
%     [amb, t, f] = amb(x, fs, nfreq)
%
%   Inputs
%     x       signal vector
%     fs      sampling frequency of x (optional, default is 1 Hz)
%     nfreq   number of samples to compute in the frequency direction, must
%             be at least 2*length(x) (optional, defaults to the next power
%             of 2 that surpasses 2*length(x))
%
%   Outputs
%     amb     ambiguity function of the signal x.
%     t       vector of sampling times (optional)
%     f       vector of frequency values (optional)

if nargin < 3
    nfreq = 2^nextpow2(2*length(x));
end

if nargin < 2
    fs = 1;
end

[tfd, ~, ~] = wigner1(x, fs, nfreq);
A = fft2(tfd)/fs;
A = flipud(fftshift(A));
n = length(x);
A = A(nfreq/4+1:nfreq/2+nfreq/4,n/2+1:n+n/2);
t = ((-n+1):n)'/fs;
f = ((-nfreq/2+1):nfreq/2)'/2/nfreq*fs;

end


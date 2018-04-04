function pamb(amb, t, f)
% pamb -- Display an image plot of an ambiguity function
%         with a linear amplitude scale.
%
%  Usage
%    ptfd(tfd, t, f, fs)
%
%  Inputs
%    amb  ambiguity function (lag along rows, doppler along columns)
%    t    vector of sampling times (optional)
%    f    vector of frequency values (optional)
%    fs   sampling frequency (optional)

% Copyright (C) -- see DiscreteTFDs/Copyright

imagesc(f,t,abs(amb)), axis('xy'), xlabel('doppler-lag'), ylabel('lag')
title('Ambiguity (abs. value)')
% figure
% imagesc(t, f, angle(amb)), axis('xy'), xlabel('doppler-lag'), ylabel('lag')
% title('Ambiguity (phase)'), colorbar
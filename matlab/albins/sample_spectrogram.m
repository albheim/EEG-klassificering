addpath('../borrowed code/');
addpath('../');

param.L = 32; param.Fs = 128; param.NFFT = 512; param.NSTEP = 1;

sub = sprintf('%02d', 5);
[X,Y,n] = aux_load('Visual',sub);
X = aux_extr(X, 769:1536);
X = aux_deci(X,4);

for i = 1:length(X)
    [S, TI, FI] = mtspectrogram(X{i}(26, :),param.L,...
                                            param.Fs,...
                                            param.NFFT,...
                                            param.NSTEP);
    c=[min(min(S)) max(max(S))];
    pcolor(TI,FI,S')  
    shading interp
    caxis(c)
    ylabel('Frequency (Hz)')
    xlabel('Time (s)')
    title('Spectrogram')
    pause
end
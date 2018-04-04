function out = aux_transform(in, method, param)
% AUX_TRANSFORM Transform EEG trials
%
% in:     cell with N-by-P data arrays  
% method: Transform method. Viable choices:
%           Spectrogram ('spec')
%           Wigner domain ('wig')
%           Ambiguity function ('amb')
%           Filtered Wigner (coming soon(TM))
%
% param:  Struct containing parameters relevant for the method.
%
% Spectrogram:  L      - Window
%               Fs     - Sampling frequency
%               NFFT   - Number of FFT points
%               NSTEP  - Number of steps to the next frame
%               wei    - Weights of MT specs (opt.)
%
% Wigner:       method - Choose amb. filter kernel.
%               par    - Choose opt. parameter (default 1)
%               Fs     - Sampling frequency
%               NFFT   - Number of FFT points
%               
% Ambiguity:    method - Choose amb. filter kernel
%               par    - Choose opt. parameter (default 1)
%               Fs     - Sampling frequency
%               NFFT   - Number of FFT points
%
% out:    3-dim output array with first dimension N, rest transf. vars

out = cell(length(in),1);
for i = 1:length(in)
    [N,P] = size(in{i});

    switch method
        case 'spec'
            out{i} = zeros(N,ceil(P/param.NSTEP),param.NFFT/2);
            if any(contains(fields(param),'wei'))
                for j = 1:N
                    out{i}(j,:,:) = mtspectrogram(in{i}(j,:)',param.L,...
                                                        param.Fs,...
                                                        param.NFFT,...
                                                        param.NSTEP,...
                                                        param.wei);
                end
            else
                for j = 1:N
                    out{i}(j,:,:) = mtspectrogram(in{i}(j,:)',param.L,...
                                                        param.Fs,...
                                                        param.NFFT,...
                                                        param.NSTEP);
                end
            end
        case 'wig'
            if ~any(contains(fields(param),'par'))
                param.par = 1;
            end
            out{i} = zeros(N,P,param.NFFT/2);
            for j = 1:N
                out{i}(j,:,:) = quadtf(in{i}(j,:)',...
                             param.method,...
                             param.par,...
                             param.Fs,...
                             param.NFFT);
            end
        case 'amb'
            if ~any(contains(fields(param),'par'))
                param.par = 1;
            end
            out{i} = zeros(N,param.NFFT,P);
            for j = 1:N
                out{i}(j,:,:) = quadamb(in{i}(j,:)',...
                             param.method,...
                             param.par,...
                             param.Fs,...
                             param.NFFT);
            end
        otherwise
            warning('Wrong type of method specified');
    end
end

end
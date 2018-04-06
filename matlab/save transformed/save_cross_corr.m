
classes = ['FA'; 'LM'; 'OB'];
for lag = [0 1 10]
    corrvecfa = zeros([31 31]);
    corrveclm = zeros([31 31]);
    corrvecob = zeros([31 31]);
    samplesfa = 0;
    sampleslm = 0;
    samplesob = 0;
    for sub = [1 2 3 4 5 6 7 8 9 11 12 13 14 15 16 17 18 19]
        [lag, sub]
        [FA, LM, OB] = load_data('Visual', sprintf('%02d', sub));
        for k = 1:length(FA.trial)
            fa = FA.trial{1, k};
            samplesfa = samplesfa + 1;
            for i = 1:31
                for j = 1:31
                    fac = crosscorr(fa(i, :), fa(j, :), lag + 1);
                    corrvecfa(i, j) = corrvecfa(i, j) + fac(1 + 1);
                end
            end
        end
        for k = 1:length(LM.trial)
            lm = LM.trial{1, k};
            sampleslm = sampleslm + 1;
            for i = 1:31
                for j = 1:31
                    lmc = crosscorr(lm(i, :), lm(j, :), lag + 1);
                    corrveclm(i, j) = corrveclm(i, j) + lmc(1 + 1);
                end
            end
        end
        for k = 1:length(OB.trial)
            ob = OB.trial{1, k};
            samplesob = samplesob + 1;
            for i = 1:31
                for j = 1:31
                    obc = crosscorr(ob(i, :), ob(j, :), lag + 1);
                    corrvecob(i, j) = corrvecob(i, j) + obc(1 + 1);
                end
            end
        end
    end
    corrvec = [corrvecfa/samplesfa corrveclm/sampleslm corrvecob/samplesob];
    labels = data(1).label;
    for c = 1:3
        figure;
        hold on;
        for i = 1:31
            subplot(6, 6, i);
            bar(corrvec(i, 31 * (c - 1) + 1:31 * c));
            set(gca, 'xlim', [1 31]);
            set(gca, 'ylim', [-1 1]);
            title(sprintf('%s lag %d %s', labels{i}, lag, classes(c)));
        end
    end
end

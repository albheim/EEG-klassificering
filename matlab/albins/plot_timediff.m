clc; close all;

Z = csvread('time_bins_classification_csp.csv');


binsize = 40;
nbins = 6;
starttime = 700;
endtime = 1600;

pred_times = starttime:binsize:(endtime - nbins * binsize);
pred_times = (pred_times - 768 + nbins * binsize / 2) * 1000 / 512;
[X, Y] = meshgrid(pred_times, pred_times);

colormap(gca,parula)
pcolor(X,Y,Z);
title('accuracy on binned retrieval data for classifiers trained on binned encoding data','interpreter','latex')
xlabel('midpoint of bin of retrieval data in ms after onset','interpreter','latex')
ylabel('midpoint of bin of encoding data in ms after onset','interpreter','latex')
hcb = colorbar
ylabel(hcb, 'Classification accuracy on test','interpreter','latex')

figure;
mean(Z(:))
hist(Z(:));
xlabel('Classification accuracy','interpreter','latex')
ylabel('Number of occurences','interpreter','latex')
title('Histogram of classification accuracies','interpreter','latex')
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 7 5];
fig.PaperPositionMode = 'manual';
print('classhist','-depsc')

figure;
accs = [0.74314815 0.7747807 0.79133691 0.78550682 0.72987005 0.68443308 0.64151235 0.62741553 0.59190709 0.57746751 0.56475309 0.54485218 0.52570176 0.52210202 0.50839344 0.47806043 0.47177551];
ts = 700:40:1360;
ts = (ts - 768 + 120) / 0.512;
plot(ts, accs)
title('validation accuracy when training on binned data','interpreter','latex')
xlabel('midpoint of bin in ms after onset (binsize=469 ms)','interpreter','latex')
ylabel('accuracy','interpreter','latex')
ylim([0 1]);
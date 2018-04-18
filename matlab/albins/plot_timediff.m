clc; close all;

Z = csvread('timepoints_sub5_18reps.csv');

binsize = 40;
nbins = 6;
starttime = 700;
endtime = 1600;

pred_times = starttime:binsize:(endtime - nbins * binsize);
pred_times = (pred_times - 768) * 1000 / 512;
[X, Y] = meshgrid(pred_times, pred_times);

colormap(gca,parula)
pcolor(X,Y,Z);
title('test accuracy in time bins for classifiers trained on time bins')
xlabel('prediction time for test in ms after onset')
ylabel('training time for study in ms after onset')
colorbar

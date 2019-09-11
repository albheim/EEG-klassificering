clear; clc; clf
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultColorbarTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultTextInterpreter','latex');
set(groot, 'defaultAxesFontSize', 15);

addpath('../');
addpath('../borrowed code/suplabel')

[X, Y, n] = aux_load('Visual', '05');

x = X{5};
t = linspace(-1.5, 2.5, 2049);
channels = [1 5 18 19 29];
labels = {'Fp1';'Fp2';'F7';'F3';'Fz';'F4';'F8';'FC5';'FC1';'FCz';'FC2';'FC6';'T7';'C3';'Cz';'C4';'T8';'CP5';'CP1';'CP2';'CP6';'P7';'P3';'Pz';'P4';'P8';'PO9';'O1';'O2';'PO10';'Iz'};
for i = 1:31
    subplot(4, 8, i)
    plot(t, x(i, :))
    ylim([-25 25])
    xlim([-1.5, 2.5])
    title(labels{i}, 'fontsize', 15);
end
suplabel('Subject 5, single trial', 't', [.08 .10 .84 .84]);
suplabel('Time relative to onset [s]', 'x');
suplabel('Mean-shifted voltage [$\mu$V]', 'y');

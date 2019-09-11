set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultColorbarTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultTextInterpreter','latex');

z = csvread('filters.csv');
figure;
for i = 1:10
    subplot(10, 1, i)
    plot(z(:, i))
    if i == 1
        title('10 filters of length 64 for channel P8 on subject 5');
    end
    ylim([-0.15 0.15])
    xlim([0 64])
end
xlabel('Samples')
close all;
m = load('out_ch6.mat');
[l, w, q] = size(m.out{1, 1});
avg = zeros(5, l);
for i = 1:18
    figure
    for j = 1:5
        subplot(5, 1, j);
        hold on;
        plot(m.out{1, i}(:, 1, j));
    end
end
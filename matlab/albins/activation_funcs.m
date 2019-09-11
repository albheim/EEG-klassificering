set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultColorbarTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultTextInterpreter','latex');


relu = @(x) (x>0).*x;
lrelu = @(x) (x>0).*x + (x<0).*x*0.4;
elu = @(x) (x>0).*x + (x<0).*(exp(x)-1)*0.8;

x = linspace(-5, 3);
hold on;
plot(x, relu(x))
plot(x, lrelu(x), '.')
plot(x, elu(x), '+')
legend(["ReLU", "LReLU", "ELU"]);
xlabel('x')
ylabel('f(x)')
close all

a = single(csvread('amatrix.csv'));
b = single(csvread('bvector.csv'));

X = linsolve(a,b);

x_gpu = csvread('x_gpu.csv');
x_cpu = csvread('x_cpu.csv');

gpu_res = csvread('gpu_res.csv');
cpu_res = csvread('cpu_res.csv');

isequalAbs = @(x,y,tol) ( norm(x-y) <= tol );

tf = isequalAbs(X,x_gpu,1e-5);
for i=1: length(tf)
    if tf(i) == 0
        disp("it is error not same ");
%         return
    end
end
disp("it is same ");
      
hold on
plot(abs(X-x_gpu)./abs(X))
title("Abs error")
figure
plot(X)
title("Solution matlab")
figure
plot(x_cpu)
title("Solution CPU")
figure
plot(x_gpu)
title("Solution GPU")

figure
plot(cpu_res)
ylim([0 1.2])
xlabel('steps')
xlabel('Convergance')
title('CPU Convergance')
grid on
figure
plot(gpu_res)
ylim([0 1.2])
xlabel('steps')
xlabel('Convergance')
title('GPU Convergance')
grid on
% xxx = zeros ( 100, 1 );
% % x = jacobi1 (100,a,b,xxx);
% for i =1 : 10
%     xxx =  jacobi1 (100,a,b,xxx);
% end
% res = a*xxx
r = norm(a*x_gpu-b);
rc = norm(a*x_cpu-b);
r2 = norm(a*X-b);

fprintf("Matlab residual: %e\n", r2);
fprintf("GPU residual: %e\n", r);
fprintf("CPU residual: %e\n", rc);
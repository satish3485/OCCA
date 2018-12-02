

a = csvread('amatrix.csv');
b = csvread('bvector.csv');

X = linsolve(a,b);

x_gpu = csvread('x_gpu.csv');
x_cpu = csvread('x_cpu.csv');

isequalAbs = @(x,y,tol) ( norm(x-y) <= tol );

tf = isequalAbs(X,x_gpu,1e-3);
for i=1: length(tf)
    if tf(i) == 0
        disp("it is error not same ");
        return
    end
end
disp("it is same ");
      
hold on
plot(abs(X-x_gpu)./abs(X))
figure
plot(X)
% xxx = zeros ( 100, 1 );
% % x = jacobi1 (100,a,b,xxx);
% for i =1 : 10
%     xxx =  jacobi1 (100,a,b,xxx);
% end
% res = a*xxx

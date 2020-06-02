%%
clear all;
format long;

set(0,'defaultAxesFontSize', 14);
set(0, 'DefaultLineLineWidth', 2);

epsilon = 1e-6;
%% 
vertices = readtable('../results/vertices.csv');  % skips the first three rows of data

edges = readtable('../results/cpu_edges.csv');
x = vertices.x;
y = vertices.y;
z = vertices.z;
%% plotting vertices
figure(3);
hold on;
plot3(x, y, z, 'r*');

%% plotting edeges
for i = 1:height(edges)
    x_tmp = [edges.x1(i), edges.x2(i)];
    y_tmp = [edges.y1(i), edges.y2(i)];
    z_tmp = [edges.z1(i), edges.z2(i)];
    plot3(x_tmp, y_tmp, z_tmp, 'b');
end
saveas(gcf, '../results/icosphere.png');

%%
%tmp ops
vertices = readtable('gpu_sorted_vertices.csv');  % skips the first three rows of data

v = vertices.x + vertices.y+ vertices.z;
% diffs = v(1:end-1)-v(2:end);
% diffs(abs(diffs) < 0.009) =     0;

% writematrix(v, 'sum.csv')
plot(vertices.x);
hold on;
plot(vertices.y);
plot(vertices.z);
legend()
% sums= readtable('sums.csv');  % skips the first three rows of data
% plot(sums.sums);


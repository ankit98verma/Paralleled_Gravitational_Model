%%
clear all;
format long;

set(0,'defaultAxesFontSize', 14);
set(0, 'DefaultLineLineWidth', 2);

epsilon = 1e-6;
%% 
vertices = readtable('../results/gpu_vertices.csv');  
% vertices = readtable('../results/cpu_vertices.csv');  

edges = readtable('../results/cpu_edges.csv');
x = vertices.x;
y = vertices.y;
z = vertices.z;
%% plotting vertices
figure(3);
hold on;
plot3(x, y, z, 'r.');
%% plotting edeges
hold on;
for i = 1:height(edges)
    x_tmp = [edges.x1(i), edges.x2(i)];
    y_tmp = [edges.y1(i), edges.y2(i)]; 
    z_tmp = [edges.z1(i), edges.z2(i)];
    plot3(x_tmp, y_tmp, z_tmp, 'b');
end
saveas(gcf, '../results/icosphere.png');

%%
%tmp opsI am not generating any 
vertices = readtable('results/gpu_sorted_vertices.csv');  % skips the first three rows of data

v = vertices.x + vertices.y+ vertices.z;
figure()
plot(0:length(v)-1, v);

sums= readtable('sums.csv');  % skips the first three rows of data
figure();
plot(0:length(sums.sums)-1, sums.sums);


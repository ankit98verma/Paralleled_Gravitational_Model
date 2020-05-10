%%
clear all;
format long;

set(0,'defaultAxesFontSize', 14);
set(0, 'DefaultLineLineWidth', 2);

%%
epsilon = 1e-6;
%%      
vertices = readtable('vertices.csv');  % skips the first three rows of data
edges = readtable('edges.csv');
vertices_sph = readtable('vertices_sph.csv');
x = vertices.x;
y = vertices.y;
z = vertices.z;
%%
% histogram of lats
% figure(1);  
% histogram(vertices_sph.theta, ceil(2*pi/1e-6));
% figure(2);
% plot(vertices_sph.theta);
%%
% plotting vertices
figure(3);
% [sx, sy, sz] = sphere(100);
% surface(sx, sy, sz, 'FaceColor', 'none', 'EdgeColor', 'k');
hold on;
plot3(x, y, z, 'r*');

%%
% plotting edeges
for i = 1:height(edges)
    x_tmp = [edges.x1(i), edges.x2(i)];
    y_tmp = [edges.y1(i), edges.y2(i)];
    z_tmp = [edges.z1(i), edges.z2(i)];
    plot3(x_tmp, y_tmp, z_tmp);
end
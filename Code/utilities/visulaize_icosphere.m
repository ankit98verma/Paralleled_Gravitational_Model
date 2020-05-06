%%
clear all;
format long;

set(0,'defaultAxesFontSize', 14);
set(0, 'DefaultLineLineWidth', 1);

%%      
vertices = readtable('vertices.csv');  % skips the first three rows of data
edges = readtable('edges.csv');

x = vertices.x;
y = vertices.y;
z = vertices.z;

vertices = [x, y, z];

%%
% plotting vertices
% [sx, sy, sz] = sphere(100);
% surface(sx, sy, sz, 'FaceColor', 'none', 'EdgeColor', 'k');
hold on;
% plot3(x, y, z, 'r*');

%%
% plotting edeges
set(0, 'DefaultLineLineWidth', 2);

for i = 1:height(edges)
    x_tmp = [edges.x1(i), edges.x2(i)];
    y_tmp = [edges.y1(i), edges.y2(i)];
    z_tmp = [edges.z1(i), edges.z2(i)];
    plot3(x_tmp, y_tmp, z_tmp, 'b');
end
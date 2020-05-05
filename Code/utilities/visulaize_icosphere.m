%%
clear all;
format long;

set(0,'defaultAxesFontSize', 14);
set(0, 'DefaultLineLineWidth', 1);

%%
T = readtable('data.csv');  % skips the first three rows of data

x = T.x;
y = T.y;
z = T.z;

%%
[sx, sy, sz] = sphere(100);
surface(sx, sy, sz, 'FaceColor', 'none', 'EdgeColor', 'k');
hold on;
plot3(x, y, z);


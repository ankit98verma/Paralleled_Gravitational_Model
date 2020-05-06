%%
clear all;
format long;

set(0,'defaultAxesFontSize', 14);
set(0, 'DefaultLineLineWidth', 1);

%%
epsilon = 1e-6;
%%      
vertices = readtable('vertices.csv');  % skips the first three rows of data
edges = readtable('edges.csv');

x = vertices.x;
y = vertices.y;
z = vertices.z;

vertices = [x, y, z];

vertices_norm = zeros(size(vertices));
vertices_norm(1, :) = vertices(1, :);
counter = 1;
for i = 1:length(x)
    tmp_x = vertices(i, 1);
    tmp_y = vertices(i, 2);
    tmp_z = vertices(i, 3);
    
    add = true;
    for j = 1:counter
        tmp_x2 = abs(vertices_norm(j, 1) - tmp_x);
        tmp_y2 = abs(vertices_norm(j, 2) - tmp_y);
        tmp_z2 = abs(vertices_norm(j, 3) - tmp_z);
        t = tmp_x2 + tmp_y2 + tmp_z2;
        if(t <= 3*epsilon)
            add = false;
            break;
        end
    end
    if(add)
        counter = counter + 1;
        vertices_norm(counter, :) = vertices(i, :);
    end
end
disp(counter);
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
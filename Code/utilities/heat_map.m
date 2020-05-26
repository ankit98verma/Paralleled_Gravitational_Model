%Draw generate heat map from lat, long maps
clear all
clc
close all


fileID = fopen('../results/output_potential.mat','r');
formatSpec = '%f %f %f %f %f';
sizeA = [5 Inf];
A = fscanf(fileID,formatSpec, sizeA);
fclose(fileID);

% Convert XYZ to Lat-Long
for i=1:1:length(A)
   lat(i) = asin(A(4,i));
   long(i) = atan2(A(3,i), A(2,i));
   potential(i) = A(5,i);
end

%% Generate 2D scatter plot
figure(1)
title('Geopotential model');
hold on
grid on
xlabel('Longitude (deg)');
ylabel('Latitude (deg)');
pointsize = 15;
scatter(long*180/pi, lat*180/pi, pointsize, potential,'filled')
colorbar;
saveas(gcf, '../results/heatmap_2.png');


%% Generate 3D scatter plot
figure(2)
title('Geopotential model');
hold on
grid on
xlabel('x');
ylabel('y');
zlabel('z');
pointsize = 15;
scatter3(A(2,:),A(3,:),A(4,:), pointsize, potential,'filled')
colorbar;

hold on
X_axis = [ 0, 0, 0; 1.2, 0, 0];
Y_axis = [ 0, 0, 0; 0, 1.2, 0];
Z_axis = [ 0, 0, 0; 0,0, 1.2];
plot3(X_axis(:,1), X_axis(:,2), X_axis(:,3), 'Linewidth' ,3);
plot3(Y_axis(:,1), Y_axis(:,2), Y_axis(:,3), 'Linewidth' ,3);
plot3(Z_axis(:,1), Z_axis(:,2), Z_axis(:,3), 'Linewidth' ,3);
view(120,25);
saveas(gcf, '../results/heatmap_3.png');

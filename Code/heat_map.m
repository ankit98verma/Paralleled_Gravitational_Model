%Draw generate heat map from lat, long maps
clear all
clc

fileID = fopen('output_potential.mat','r');
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

% heatmap( potential , long, lat);
% ,'ColorVariable', cvar );



function displayLidarOverlayPointCloud(lidarPC, labelMap, classNames)
%displayLidarOverlayPointCloud Overlay labels over the point cloud. 
%
% Copyright 2026 The MathWorks, Inc.

% Load the lidar color map.
cmap = jet(length(classNames));

% Generate color array for point cloud
cmap_Array = cmap(single(labelMap),:);
ptCloud_Overlay = pointCloud(reshape(lidarPC(:,:,1:3),[],3),"Color",cmap_Array);

pcshow(ptCloud_Overlay);
view(-90,0);
helper.pixelLabelColorbar(cmap, classNames);
end

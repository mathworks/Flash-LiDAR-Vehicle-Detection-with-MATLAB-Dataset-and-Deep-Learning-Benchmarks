function displayLidarOverlayPointCloud(lidarPC, labelMap, classNames)
%displayLidarOverlayPointCloud Overlay labels over the point cloud. 
%
% Copyright 2026 The MathWorks, Inc.

% Load the lidar color map.
cmap = jet(length(classNames));

% Generate color array for point cloud
labelMap = single(labelMap);
labelMap(isnan(labelMap)) = 1;
cmap_Array = cmap(labelMap,:);
ptCloud_Overlay = pointCloud(reshape(lidarPC.Location,[],3),"Color",cmap_Array);

pcshow(ptCloud_Overlay);
view(-90,0);
helper.pixelLabelColorbar(cmap, classNames);
end

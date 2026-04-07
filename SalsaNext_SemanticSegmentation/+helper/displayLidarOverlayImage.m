function displayLidarOverlayImage(lidarImage, labelMap, classNames)
%displayLidarOverlayImage Overlay labels over the intensity image. 
%
%  displayLidarOverlayImage(lidarImage, labelMap, classNames)
%  displays the overlaid image. lidarImage is a five-channel lidar input.
%  labelMap contains pixel labels and classNames is an array of label
%  names.
%
% Copyright 2026 The MathWorks, Inc.

% Read the intensity channel from the lidar image.
intensityChannel = im2uint8(rescale(lidarImage(:,:,4)));

% Load the lidar color map.
cmap = jet(length(classNames));

% Overlay the labels over the intensity image.
B = labeloverlay(intensityChannel,labelMap,'Colormap',cmap,'Transparency',0.4);

% Resize for better visualization.
B = imresize(B, 'Scale', [10 10], 'method', 'nearest');
imshow(B);
helper.pixelLabelColorbar(cmap, classNames);
end



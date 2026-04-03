function [mask, cuboid] = convertROItoMask5Ch(Img5ch,maskSize, ROI, RangeBuffer, bgRange)
% Improved version of ROI to Mask conversion function
% This uses the same "mode" algorithm but also includes background removal, 
% ground segmentation using SMRF, point cloud denoising, point cloud clustering,
% and a cuboid fitting to also output the 3D cuboid ground truth on top of
% the segmentation mask of the vehicle. 
% Select the range channel from a 5-channel image
RangeImg = Img5ch(:,:,5);   
inROI = false(maskSize);
ROI = round(ROI);
% Mark pixels inside the provided ROI (clamped to 128x128 image)
inROI(ROI(2) :min(128,ROI(2)+ROI(4)), ROI(1):min(128,ROI(1)+ROI(3))) = true;
% Build a histogram of range values within ROI below background range (quantized to 10)
rangeWindow = fix(RangeImg(inROI & (RangeImg < bgRange))./10).*10;
% Estimate dominant range within the ROI
midRange = mode(rangeWindow, "all");
% Create a band-pass mask around the dominant range
inRange = (RangeImg > midRange-RangeBuffer) & (RangeImg < midRange+RangeBuffer);
mask = (inRange & inROI);
XYZ = Img5ch(:,:,1:3);
pCloud = pointCloud(XYZ);
% Segment ground points and exclude them from the mask
groundPtsIdx = segmentGroundSMRF(pCloud, ElevationThreshold=0.001);
mask(groundPtsIdx) = false; 
if nnz(mask) ~= 0
    goodIdx = find(mask);
    pCloudCrop = select(pCloud,goodIdx);
    % Remove sparse outliers from the cropped point cloud
    [~,~,outlierIndices] = pcdenoise(pCloudCrop, "NumNeighbors",10, "Threshold",2);
    mask(goodIdx(outlierIndices)) = false;
    
    goodIdx = find(mask);
    pCloudCrop = select(pCloud,goodIdx);
    % Segment remaining points by distance and remove small/noise clusters
    labels = pcsegdist(pCloudCrop,5);
    outlierIndices = find(labels ~= mode(labels));
    mask(goodIdx(outlierIndices)) = false;
end
goodIdx = find(mask);
pCloudCrop = select(pCloud,goodIdx);
% Fit a cuboid to the filtered point cloud
cuboid = pcfitcuboid(pCloudCrop);
end

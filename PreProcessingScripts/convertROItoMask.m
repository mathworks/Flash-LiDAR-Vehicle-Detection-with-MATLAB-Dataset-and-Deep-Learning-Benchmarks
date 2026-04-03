function [mask] = convertROItoMask(Image,ROI, RangeBuffer)
% Initialize mask
mask = false(size(Image));   

if ~isempty(ROI)
    % Create mask for all points inside bounding box
    inROI = false(size(Image));
    ROI = round(ROI);
    inROI(ROI(2) :min(128,ROI(2)+ROI(4)), ROI(1):min(128,ROI(1)+ROI(3))) = true;
    
    % Calculate the "mode" of the range of the points within the bounding
    % box, which is an estimate of the distance of the detected vehicle
    % and eliminate all points that are too far or too close compared to the 
    % estimated distance (Range +/- threshold 
    rangeWindow = fix(Image(inROI & Image~=0)/1000)*1000;
    midRange = mode(rangeWindow, "all"); 
    inRange = (Image > midRange-RangeBuffer) & (Image < midRange+RangeBuffer);
    
    % Combine logical masks
    mask = (inRange & inROI);
    
    % Morphological operations to improve estimated masks
    % Fill holes
    mask = imfill(mask, 'holes');
    % Close mask
    radius = 9;
    decomposition = 0;
    se = strel('disk', radius, decomposition);
    mask = imclose(mask, se);
end

end


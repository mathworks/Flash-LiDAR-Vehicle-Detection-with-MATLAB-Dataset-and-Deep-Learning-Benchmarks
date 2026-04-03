function [data] = augmentData(A)
% The augmentDataForPCBDefectDetection function randomly applies horizontal
% flipping and scaling to pairs of images and bounding boxes. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25.
%
% Copyright 2026 The MathWorks, Inc.

data = cell(size(A));
for ii = 1:size(A,1)
    I = A{ii,1};
    bboxes = A{ii,2};
    labels = A{ii,3};
    sz = size(I);

    % Randomly flip image.
    tform = randomAffine2d(XReflection=true,YReflection=false,Scale=[1 1.1],XTranslation=[-10 10],YTranslation=[-10 10]);
    rout = affineOutputView(sz,tform,BoundsStyle="centerOutput");
    I = imwarp(I,tform,OutputView=rout);
    
    % Apply same transform to boxes.
    [bboxes,indices] = bboxwarp(bboxes,tform,rout,OverlapThreshold=0.25);
    labels = labels(indices);
    
    % Return original data only when all boxes are removed by warping.
    if isempty(indices)
        data(ii,:) = A(ii,:);
    else
        data(ii,:) = {I,bboxes,labels};
    end
end
end
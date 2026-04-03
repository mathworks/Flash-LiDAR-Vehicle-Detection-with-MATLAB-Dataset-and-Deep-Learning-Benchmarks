%% Train SalsaNext Model for Vehicle Segmentation with Flash LiDAR Data

%% Load Pretrained Model
net = load('SalsaNext_Randomized.mat').net; 
% net = load('SalsaNext_Pretrained.mat').net; %If using PandaSet pre-training

%% Load Datastores
labelsFolder = "..\..\Masks"; %Switch to directory where the data was downloaded
imagesFolder = "..\..\Images_5Ch"; %Switch to directory where the data was downloaded

imds = imageDatastore(imagesFolder, ...
    'FileExtensions', '.mat', ...
    'ReadFcn', @helper.imageMatReader, ...
    'IncludeSubfolders',true);
 

classNames = ["Undefined"
              "Background"
              "ATV"
              "JeepGreen"
              "PickupWhite"
              "SUVBlack"
              "VanWhite"
              "Plane"
              "SedanBlack"
              "SportsCarYellow"];

numClasses = numel(classNames);

% Specify label IDs from 0 to the number of classes.
labelIDs = 0 : numClasses-1;
pxds = pixelLabelDatastore(labelsFolder, classNames, labelIDs, "IncludeSubfolders",true);

%% Prepare Training, Validation, and Test Sets
% Use the partitionLidarData helper function to split the data into
% training, images, respectively.
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = helper.partitionLidarData(imds, pxds);

dsTrain = combine(imdsTrain,pxdsTrain);
dsVal = combine(imdsVal,pxdsVal);

%% Configure Pretrained Network
% Changing input size to size of flash lidar data and output size to 
% required number of classes.
inputSize = [128, 128, 5];
net = replaceLayer(net, 'Input_input.1', imageInputLayer(inputSize, 'Name', 'Input_input.1', 'Normalization', 'none'));
net = replaceLayer(net, 'Conv_191', convolution2dLayer([1,1], numClasses, 'Name', 'Conv_191'));

%% Define training options. 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',1,...
    'LearnRateDropFactor',0.1,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...
    'ValidationData',dsVal,...
    'MaxEpochs',30, ...  
    'MiniBatchSize',30, ... %Adjust based on GPU memory
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 4, ...
    'ValidationFrequency', 400);


%% Train the network
doTraining = false;

if doTraining
    % Set weights to give more importance to vehicle classes (#3-10) over
    % background and undefined classes (#1-2)
    weights = [0.02 0.02 0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.12]; 
    [trainedNet,info] = trainnet(dsTrain,net, @(Y,T)crossentropy(Y,T,weights,WeightsFormat="UC"),options);
    modelName = "SalsaNext_Trained_" +string(datetime)+ ".mat";
    save(modelName, 'trainedNet');
else
    load("SalsaNext_Trained.mat","trainedNet");
end

%% Test SalsaNext Segmentation Model
% Run model on a single test image
testImg = read(imdsTest);
testLabels = read(pxdsTest);
predictedResult = semanticseg(testImg,trainedNet);
tiledlayout(1, 2);
% Display Ground Truth
nexttile
helper.displayLidarOverlayImage(testImg, testLabels{1}, classNames);
title('Semantic Segmentation Ground Truth');
% Display Predicted Output
nexttile
helper.displayLidarOverlayImage(testImg, predictedResult, classNames);
title('Semantic Segmentation Prediction');
% reset(imdsTest); reset(pxdsTest)%To restart from first image

%% Run model on Entire Test data
testResults = semanticseg(imdsTest, trainedNet, "Classes", classNames);

%%  Evaluate metrics
metrics = evaluateSemanticSegmentation(testResults,pxdsTest);

% Visualize per class metrics
metrics.ClassMetrics

%% Visualize results of test data - Image Format
figure('Position', [50 50 1800 900])
reset(pxdsTest)
reset(imdsTest)

while hasdata(imdsTest)
    testImg = read(imdsTest);
    testLabels = read(pxdsTest);
    predictedResult = semanticseg(testImg,trainedNet);
    
    tiledlayout(1, 2);
    % Display Ground Truth 
    nexttile
    helper.displayLidarOverlayImage(testImg, testLabels{1}, classNames);
    title('Semantic Segmentation Ground Truth');
    % Display Predicted Output 
    nexttile
    helper.displayLidarOverlayImage(testImg, predictedResult, classNames);
    title('Semantic Segmentation Prediction');
    drawnow
    pause(0.1)
end

%% Visualize results of test data - Point Cloud Format
figure('Position', [50 50 1800 900])
reset(pxdsTest)
reset(imdsTest)
cmap = helper.lidarColorMap();

count = 0;
while hasdata(imdsTest)
    count = count + 1;
    testImg = read(imdsTest);
    testLabels = read(pxdsTest);
    predictedResult = semanticseg(testImg,trainedNet);
    
    % Load the un-normalized data for display
    testImg = load(imdsTest.Files{count}).Img5ch;
    
    tiledlayout(1, 2);
    % Display Ground Truth 
    colormap_GroundTruth = cmap(single(testLabels{1}),:);
    ptCloud_GroundTruth = pointCloud(reshape(testImg(:,:,1:3),[],3),"Color",colormap_GroundTruth);
    nexttile
    pcshow(ptCloud_GroundTruth);
    view(-90,0)
    title('Semantic Segmentation Ground Truth');
    % Display Predicted Output 
    colormap_Predicted = cmap(single(predictedResult),:);
    ptCloud_Predicted = pointCloud(reshape(testImg(:,:,1:3),[],3),"Color",colormap_Predicted);
    nexttile
    pcshow(ptCloud_Predicted);
    view(-90,0)
    title('Semantic Segmentation Prediction');

    drawnow
    pause(0.1)
end
% Copyright 2026 The MathWorks, Inc.
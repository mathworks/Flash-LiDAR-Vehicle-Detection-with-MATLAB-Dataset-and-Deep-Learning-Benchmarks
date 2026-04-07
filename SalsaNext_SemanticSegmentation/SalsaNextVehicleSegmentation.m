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
 

classNames = ["NoReturn"
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

% Use the combine function to combine the images and labels into a single 
% datastore for training. Then, separate the data into training, validation, and test data. 
cds = combine(imds,pxds);
skipFrame = 100; %Reduce training data by skipping #frames
numImages = cds.numpartitions;
Indices = 1:skipFrame:numImages;

% Use 70% for training, 10% for validation, and the rest (20%) for testing
rng("default");
numIndices = length(Indices);
shuffleOrder = randperm(numIndices);
shuffledIndices = Indices(shuffleOrder);
numTrain = floor(0.7*numIndices);
numVal = ceil(0.1*numIndices);

cdsTrain = subset(cds,shuffledIndices(1:numTrain));
cdsVal = subset(cds,shuffledIndices(numTrain+1:numTrain+numVal));
cdsTest = subset(cds,shuffledIndices(numTrain+numVal+1:end));

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
    'ValidationData',cdsVal,...
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
    weights = [0.1 0.1 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8]; 
    [trainedNet,info] = trainnet(cdsTrain,net, @(Y,T)crossentropy(Y,T,weights,WeightsFormat="UC"),options);
    modelName = "SalsaNext_Trained_" + datetime('now', Format='yyyy_MM_dd_HH_mm') + ".mat";
    save(modelName, 'trainedNet');
else
    load("SalsaNext_Trained.mat","trainedNet");
end

%% Test SalsaNext Segmentation Model
% Run model on a single test image
testCell = read(cdsTest);
testImg = testCell{1};
testLabels = testCell{2};
predictedResult = semanticseg(testImg,trainedNet);
tiledlayout(1, 2);
% Display Ground Truth
nexttile
helper.displayLidarOverlayImage(testImg, testLabels, classNames);
title('Semantic Segmentation Ground Truth');
% Display Predicted Output
nexttile
helper.displayLidarOverlayImage(testImg, predictedResult, classNames);
title('Semantic Segmentation Prediction');
% reset(cdsTest); %To restart from first image

%% Run model on Entire Test data
imdsTest = cdsTest.UnderlyingDatastores{1};
testResults = semanticseg(imdsTest, trainedNet, "Classes", classNames);

%%  Evaluate metrics
pxdsTest = cdsTest.UnderlyingDatastores{2};
metrics = evaluateSemanticSegmentation(testResults,pxdsTest);

% Visualize per class metrics
metrics.ClassMetrics

%% Visualize results of test data - Image Format
figure('Position', [50 50 1800 900])
reset(cdsTest)

while hasdata(imdsTest)
    testCell = read(cdsTest);
    testImg = testCell{1};
    testLabels = testCell{2};
    predictedResult = semanticseg(testImg,trainedNet);
    
    tiledlayout(1, 2);
    % Display Ground Truth 
    nexttile
    helper.displayLidarOverlayImage(testImg, testLabels, classNames);
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
reset(cdsTest)
imdsTest = cdsTest.UnderlyingDatastores{1};

count = 0;
while hasdata(imdsTest)
    count = count + 1;
    testCell = read(cdsTest);
    testImg = testCell{1};
    testLabels = testCell{2};
    predictedResult = semanticseg(testImg,trainedNet);
    
    % Load the un-normalized data for display
    testPC = load(imdsTest.Files{count}).Img5ch;
    
    tiledlayout(1, 2);
    % Display Ground Truth 
    nexttile
    helper.displayLidarOverlayPointCloud(testPC, testLabels, classNames)
    title('Semantic Segmentation Ground Truth');
    % Display Predicted Output 
    nexttile
    helper.displayLidarOverlayPointCloud(testPC, predictedResult, classNames)
    title('Semantic Segmentation Prediction');

    drawnow
    pause(0.1)
end
% Copyright 2026 The MathWorks, Inc.
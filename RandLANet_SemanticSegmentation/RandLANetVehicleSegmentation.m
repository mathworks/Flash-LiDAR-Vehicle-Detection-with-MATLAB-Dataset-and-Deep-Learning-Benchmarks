%% Train RandLANet Model for Vehicle Segmentation with Flash LiDAR Data

%% Load Datastores
dataFolder = "..\..\PointClouds"; %Switch to directory where the data was downloaded
labelFolder = "..\..\Masks"; %Switch to directory where the data was downloaded

pcds = fileDatastore(dataFolder,"ReadFcn",@(x) pcread(x),'IncludeSubfolders',true);

classNames = ["unlabelled"
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
labelIDs = 1 : numClasses;
pxds = pixelLabelDatastore(labelFolder,classNames,labelIDs, "IncludeSubfolders",true);

%% Prepare Training, Validation, and Test Sets
% Use the combine function to combine the point clouds and labels into a single 
% datastore for training. Then, separate the data into training, validation, and test data. 
cds = combine(pcds,pxds);
skipFrame = 100; %Reduce training data by skipping frames
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

%% Display sample point cloud and ground truth masks
trainCell = read(cdsTrain);
trainPC = trainCell{1};
trainMask = trainCell{2};

% Format mask to display as unique colors
helper.displayLidarOverlayPointCloud(trainPC, trainMask, classNames)
% reset(cdsTrain) %To restart from first image

%% Use the randlanet function to create a RandLANet segmentation network.
net = randlanet("none",classNames,GridStep = 0.1, NumPoints=16384, PointProperty="intensity");
% net = randlanet("pandaset",classNames,GridStep = 0.1, NumPoints=16384,PointProperty="intensity"); %If using PandaSet pre-training

%% Define Training Options
% Use the Adam optimization algorithm to train the network. 
% Use the trainingOptions function to specify the hyperparameters.
learningRate = 0.001;
numEpochs = 10;
miniBatchSize = 20; %Adjust based on GPU memory
learnRateDropFactor = 0.5;
executionEnvironment = "auto";
preprocessEnvironment = "serial";

options = trainingOptions("adam", ...
    InitialLearnRate = learningRate, ...
    MaxEpochs = numEpochs, ...
    MiniBatchSize = miniBatchSize, ...
    LearnRateSchedule = "piecewise", ...
    LearnRateDropPeriod = 1,...
    LearnRateDropFactor = learnRateDropFactor, ...
    Plots = "training-progress", ...
    ExecutionEnvironment = executionEnvironment, ...
    PreprocessingEnvironment = preprocessEnvironment, ...
    ResetInputNormalization = false, ...
    CheckpointFrequencyUnit = "epoch", ...
    CheckpointFrequency = 1, ...
    CheckpointPath = tempdir, ...
    BatchNormalizationStatistics="moving",...
    verbose = true, ...
    ValidationData = cdsVal, ...
    ValidationFrequency = 500);

%% Train the Networks
% Use the trainRandlanet function to train the RandLANet segmentation network, set the doTraining argument to true. Otherwise, load a pretrained segmentation network.
doTraining = false;

if doTraining
    trainedNet = trainRandlanet(cdsTrain,net,options);
    modelName = "RandLANet_Trained_" + datetime('now', Format='yyyy_MM_dd_HH_mm') + ".mat";
    save(modelName, "trainedNet");
else
    load("RandLANet_Trained.mat");
end

%% Test RandLANet Segmentation Model
% Run model on a single test image
testCell = read(cdsTest);
testPC = testCell{1};
testMask = testCell{2};
testPred = segmentObjects(trainedNet,testPC);

tiledlayout(1, 2);
% Display Ground Truth
nexttile
helper.displayLidarOverlayPointCloud(testPC, testMask, classNames)
title('Semantic Segmentation Ground Truth');
% Display Predicted Output
nexttile
helper.displayLidarOverlayPointCloud(testPC, testPred, classNames)
title('Semantic Segmentation Prediction');
% reset(cdsTest); %To restart from first image

%% Run model on Entire Test data
pcdsTest = cdsTest.UnderlyingDatastores{1};
testResults = segmentObjects(trainedNet,pcdsTest);

%%  Evaluate metrics
pxdsTestFiles = sort(cdsTest.UnderlyingDatastores{2}.Files);
pxdsTest = pixelLabelDatastore(pxdsTestFiles,classNames,labelIDs);
metrics = evaluateSemanticSegmentation(testResults,pxdsTest);

% Visualize per class metrics
metrics.ClassMetrics

%% Visualize results of test data
figure('Position', [50 50 1800 900])
reset(cdsTest)

while hasdata(cdsTest)
    testCell = read(cdsTest);
    testPC = testCell{1};
    testMask = testCell{2};
    testPred = segmentObjects(trainedNet,testPC);

    tiledlayout(1, 2);
    % Display Ground Truth
    nexttile
    helper.displayLidarOverlayPointCloud(testPC, testMask, classNames)
    title('Semantic Segmentation Ground Truth');
    % Display Predicted Output
    nexttile
    helper.displayLidarOverlayPointCloud(testPC, testPred, classNames)
    title('Semantic Segmentation Prediction');
   
    drawnow; % Update the figure
    pause(0.1)
end

% Copyright 2026 The MathWorks, Inc.
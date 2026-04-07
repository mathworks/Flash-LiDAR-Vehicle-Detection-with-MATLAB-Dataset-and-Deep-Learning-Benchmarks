%% Train YOLO-X Model for Vehicle Detection with Flash LiDAR Data

%% Load Datastores
labelsFile = "..\..\TrainingTable_BoundingBoxes.mat"; %Switch to directory where the data was downloaded
dataFolder = "..\..\Images_2Ch"; %Switch to directory where the data was downloaded
addpath(genpath(dataFolder))
TrainingTable = load(labelsFile).TrainingTable;

% Balance the dataset by taking only a certain number of labels per class
nLabels = 2500; %number of labels per class
vehicleDataset = helper.balanceDataset(TrainingTable, nLabels);

% Create image and bounding box datastores
imds = imageDatastore(vehicleDataset{:,"imageFilename"});
blds = boxLabelDatastore(vehicleDataset(:,2:end));
% Combine them for training
cds = combine(imds, blds);

% Visualize label distribution
tbl = countEachLabel(blds);
figure
bar(tbl.Label,tbl.Count)
ylabel("Frequency")


%% Split the data set into training, validation, and test sets. 
% Because the total number of images is relatively small, allocate a relatively 
% large percentage (70%) of the data for training. Allocate 10% for validation 
% and the rest (20%) for testing.
rng("default");
numImages = cds.numpartitions;
numTrain = floor(0.7*numImages);
numVal = floor(0.1*numImages);

shuffledIndices = randperm(numImages);
trainingData = subset(cds,shuffledIndices(1:numTrain));
validationData = subset(cds,shuffledIndices(numTrain+1:numTrain+numVal));
testData = subset(cds,shuffledIndices(numTrain+numVal+1:end));


%% Augment the training dataset
% Add reflacion, translation and other transforms to diversify the images
trainingData = transform(trainingData,@helper.augmentData);

% Display one of the training images and box labels to verify augmentation
data = read(trainingData);
I = data{1};
bbox = data{2};
label = data{3};

annotatedImage = insertObjectAnnotation(I,"Rectangle",bbox, label);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%% Create a YOLOX Object Detector Network
% Specify the network input size to be used for training. 
inputSize = [128 128 3];
% Specify the name of the object classes to detect.
className = {'ATV', 'Jeep', 'PickupWhite', 'SUVBlack', 'VanWhite', 'Plane', 'SedanBlack', 'SportsCar'};
% Create the YOLO X object detector by using the yolovxObjectDetector function. 
% specify the name of the pretrained YOLO X detection network trained on COCO dataset. 
% Specify the class name and the image input size.
YOLODetector = yoloxObjectDetector("small-coco",className,InputSize=inputSize);

%If you need a smaller model for deployment use the "tiny" architecture
% YOLODetector = yoloxObjectDetector("tiny-coco",className,InputSize=inputSize); 

%% Specify Training Options
% Use trainingOptions to specify network training options. 
% Train the object detector using the Adam solver for 70 epochs with a constant learning rate 0.001. 
% "ResetInputNormalization" should be set to false and "BatchNormalizationStatistics" 
% should be set to "moving". Set "ValidationData" to the validation data and "ValidationFrequency" 
% to 1000. To validate the data more often, you can reduce the "ValidationFrequency" 
% which also increases the training time. Use "ExecutionEnvironment" to determine 
% what hardware resources will be used to train the network. Default value for this 
% is "auto" which selects a GPU if it is available, otherwise selects the CPU. 
% Set "CheckpointPath" to a temporary location. This enables the saving of partially 
% trained detectors during the training process. If training is interrupted, such as 
% by a power outage or system failure, you can resume training from the saved checkpoint. 

options = trainingOptions("adam",...
    GradientDecayFactor=0.9,...
    SquaredGradientDecayFactor=0.999,...
    InitialLearnRate=0.001,...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.98, ...
    LearnRateDropPeriod=1, ... 
    MiniBatchSize=128,... %Adjust based on GPU memory
    L2Regularization=0.0005,...
    MaxEpochs=30,...
    BatchNormalizationStatistics="moving",...
    DispatchInBackground=false,... % only with multiple GPU
    ResetInputNormalization=false,...
    Shuffle="every-epoch",...
    VerboseFrequency=20,...
    ValidationFrequency=20,...
    CheckpointPath=tempdir,...
    ValidationData=validationData, ...
    Plots="training-progress"); %Plots="training-progress"

%% Train YOLOX Object Detector
% Use the trainYOLOXObjectDetector function to train YOLOX object detector. 
% This example is run on an NVIDIA™ RTX 3080Ti GPU with 12 GB of memory. 
% Training this network took approximately 3 hours using this setup. 
% The training time will vary depending on the hardware you use. Instead of 
% training the network, you can also use a pretrained YOLO X object detector 
% in the Computer Vision Toolbox ™. 
doTraining = false;
if doTraining        
    % Train the YOLO-X detector.
    [YOLODetector,info] = trainYOLOXObjectDetector(trainingData,YOLODetector,options);
    modelName = "YOLOXDetector_Trained_" + datetime('now', Format='yyyy_MM_dd_HH_mm') + ".mat";
    save(modelName, 'YOLODetector');
else
    % Load pretrained detector for the example.
    load("YOLOXDetector_Trained.mat");
end


%% Test YOLOX Object Detector
% Run the detector on a single test image
data = read(testData);
[bboxPred,scorePred,labelPred] = detect(YOLODetector,data{1});
annotatedImage = helper.addDetectionAnnotation(data, bboxPred, labelPred, scorePred);

figure
imshow(annotatedImage)
% reset(testData); %To restart from first image

%% Evaluate Detector on Entire Test Set
% Evaluate the trained object detector on a large set of images to measure the performance. 
% Computer Vision Toolbox™ provides object detector evaluation functions (evaluateObjectDetection) 
% to measure common metrics such as average precision and log-average miss rates. 
% For this example, use the average precision metric to evaluate performance. 
% The average precision provides a single number that incorporates the ability 
% of the detector to make correct classifications (precision) and the ability 
% of the detector to find all relevant objects (recall).

% Run the detector on all the test images.
detectionResults = detect(YOLODetector,testData);

%% Evaluate the results
metrics = evaluateObjectDetection(detectionResults,testData);

%% Display evaluation metrics
summarize(metrics)
metrics.ClassMetrics(:,1:3)

allLabels = vertcat(detectionResults{:,3}{:});
allScores = vertcat(detectionResults{:,2}{:});

figure
for ii = 1:height(metrics.ClassMetrics)
    currClass = className(ii);
    currClassScores = allScores(allLabels == currClass);
    currClassScores = [1;sort(currClassScores,'descend')];
    p = plot(metrics.ClassMetrics{ii,'Recall'}{:},metrics.ClassMetrics{ii,'Precision'}{:});
    p.DataTipTemplate.DataTipRows(end+1) = dataTipTextRow('Score', double(currClassScores));
    hold on
end
hold off
xlabel("Recall")
ylabel("Precision")
title("Precision vs Recall For All Classes")
legend(string(className)' + " avgP: " + metrics.ClassMetrics{:,'AP'}, "Location", "southwest")
grid on

figure
confusionchart(metrics.ConfusionMatrix, [categorical(className)'; {'Undetected'}])
figure
confusionchart(metrics.ConfusionMatrix, [categorical(className)'; {'Undetected'}], "Normalization", "column-normalized")

%% Visualize the results on the test data
reset(testData);
figure
for i = 1:height(detectionResults)
    data = read(testData);
    [bboxPred,scorePred,labelPred] = detect(YOLODetector,data{1});

    annotatedImage = helper.addDetectionAnnotation(data, bboxPred, labelPred, scorePred);
    
    imshow(annotatedImage)
    pause(1)
end 

% Copyright 2026 The MathWorks, Inc.


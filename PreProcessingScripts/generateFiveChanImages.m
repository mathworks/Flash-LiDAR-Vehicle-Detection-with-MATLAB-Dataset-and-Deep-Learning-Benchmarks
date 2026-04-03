function generateFiveChanImages(dataFolder,imagesFolder)
% Function that describes the creation of the 5 Channel images used to
% train the SalsaNext model

if ~exist(imagesFolder,'dir')
    mkdir(imagesFolder);
end

rangeFolder = dir(fullfile(dataFolder,'*.png'));
numFiles = size(rangeFolder,1);
lowNoise = 1;
highNoise = 1500;
highInt = 3200;

for ii = 1:numFiles
    % Load images and calculate xyz
    Img = im2double(imread(fullfile(dataFolder, rangeFolder(ii).name)));
    Range = Img(:,:,1).* highNoise;
    Intensity = Img(:,:,3) .* highInt;

    % Set outliers to NaN
    OutLog = Range < lowNoise | Range > highNoise;
    Range(OutLog) = NaN;
    Intensity(OutLog) = NaN;
    XYZ = calculateXYZ(Range);

    % Image are of 5-channels, namely x,y,z,intensity and range.
    Img5ch = zeros([size(Img,[1,2]), 5]);
    Img5ch(:,:,1:3) = XYZ;
    Img5ch(:,:,4) = Intensity;
    Img5ch(:,:,5) = Range;
    
    
    % Store images and labels as .mat and .png files respectively.
    imfile = fullfile(imagesFolder,[rangeFolder(ii).name(1:end-4), '.mat']);
    save(imfile,'Img5ch');   
end
end
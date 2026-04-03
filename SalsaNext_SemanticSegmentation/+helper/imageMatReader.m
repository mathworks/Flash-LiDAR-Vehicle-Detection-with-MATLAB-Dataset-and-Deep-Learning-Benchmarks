function data = imageMatReader(filename)
%imageMatReader Reads custom MAT files containing 5-channel Flash Lidar data.

%  DATA = imageMatReader(FILENAME) returns the 5 channel image saved in FILENAME.

% Copyright 2026 The MathWorks, Inc

% Load file and extract array
d = load(filename);
f = fields(d);
data = d.(f{1});
% Replace invalid values with zeros
index = isnan(data);
data(index) = 0;

data(:,:,1) = rescale(data(:,:,1), 0, 1, "InputMin", 0, "InputMax", 1500);
data(:,:,2) = rescale(data(:,:,2), 0, 1, "InputMin", -40, "InputMax", 40);
data(:,:,3) = rescale(data(:,:,3), 0, 1, "InputMin", -40, "InputMax", 40);
data(:,:,4) = rescale(data(:,:,4), 0, 1, "InputMin", 0, "InputMax", 3200);
data(:,:,5) = rescale(data(:,:,5), 0, 1, "InputMin", 0, "InputMax", 1500);
end
    
    
    
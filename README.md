# Flash LiDAR Vehicle Detection with MATLAB Dataset and Deep Learning Benchmarks

![Flash Lidar Vehicle Tracking](FlashLidarGIF.gif)

## Point Cloud Processing, Dataset Download, Deep Learning Models
[![View <File Exchange Title> on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/182842-flash-lidar-vehicle-detection) 

This repository provides code and workflows to test several state-of-the-art vehicle detection deep learning algorithms —including YOLOX, SalsaNext, and RandLA-Net— on a Flash Lidar dataset. The models are applied to 2D, 3D, and 5-channel data, supporting comprehensive benchmarking and research in autonomous vehicle perception. Also included are some point cloud processing techniques applied to the Flash Lidar to perform operations such as converting the raw data to images and point clouds, converting the ground truth bounding boxes to 2D/3D segmentation masks and cuboids, and more.  

## Dataset Description
This repository uses a Public Domain dataset collected specifically for benchmarking vehicle detection algorithms using a Flash LiDAR sensor. The dataset consists of 35 sequences featuring eight different vehicles—including seven four-wheel vehicles and a small plane. Each sequence was captured using a 128x128 Flash LiDAR sensor, recording range and intensity information for every detected point.

The dataset is available in four distinct formats to support a variety of deep learning models:

### 2D Image Format:

Each image is a uint16 3 channel image, where range data is mapped to the 'red' channel and intensity data to the 'blue' channel. This format is recommended to test 2D object detection models such as YOLOX and other 2D segmentation models. Included with the images is the ground truth in form of bounding boxes and segmentation masks.

### Point Cloud Format:

Data is provided as PCD files, retaining both the spatial location and intensity for each point.
Ideal for 3D detection and segmentation models like RandLA-Net. Included with the point clouds is the ground truth in form of cuboids and segmentation masks.


### Multi-Channel Image Format:

Each 128x128 image contains five channels: X, Y, Z, Range, and Intensity.
Used for multi-channel models such as SalsaNext. Included with the images is the ground truth in form of bounding boxes, cuboids, and segmentation masks.


### Raw Format:

Two arrays of size 128x128xnFrames for each sequence, representing the unfiltered and unprocessed range and intensity recorded directly from the sensor. Included with the raw data is the ground truth in form of bounding boxes, cuboids, and segmentation masks.


All data has been denoised by removing points beyond the 99th percentile of the standard deviation, and values are normalized and scaled to maximize detail in each format (e.g., stretching uint16 values to the full 0–2<sup>16</sup> range).


## Setup 
To Run:
1. Download the Flash Lidar dataset from the links below in the format that works with the model that you want to test.
2. Clone this repository and open it in MATLAB®.
3. Extract the data and store it in the "Data\" directory of the repo.
3. Follow the installation instructions below to set up required products and dependencies.
4. Run the example scripts according to the model that you want to test.

### Flash Lidar Dataset Download Links

2D Data (PNG images + bounding box + mask ground truth)
Download 2D Data (https://ssd.mathworks.com/supportfiles/lidar/data/FlashLidar/Data_2D.tar)

3D Data (PCD Point cloud + cuboids + mask ground truth)
Download 3D Data (https://ssd.mathworks.com/supportfiles/lidar/data/FlashLidar/Data_3D.tar)

5-Channel Data (MAT 5-channel images [x, y, z, range, intensity] + bounding box + cuboids + mask ground truth)
Download 5-Channel Data (https://ssd.mathworks.com/supportfiles/lidar/data/FlashLidar/Data_5Ch.tar)

Raw Data (MAT arrays [range, intensity] + bounding box + cuboids + mask ground truth)
Download Raw Data (https://ssd.mathworks.com/supportfiles/lidar/data/FlashLidar/Data_Raw.tar)

Note that this dataset is very large and it can take several minutes to hours, depending on internet speed, to download the compressed folders using the links above.

### Additional information about set up
In order to run the examples in this repo you will need to install the MATLAB® products listed below.
Additionally you will need to install the support package for each deep learning model that you want to train or test using the provided dataset. 

Note that to run and train the deep learning models it is recommended to use a computer with a dedicated GPU. 

 
### MathWorks Products (https://www.mathworks.com)

- MATLAB®
- Deep Learning Toolbox™
- Computer Vision Toolbox™
- Lidar Toolbox™
- Automated Visual Inspection Library™
- Parallel Computing Toolbox™ (Recommended)

Requires MATLAB release R2024A or newer


## Getting Started 

This repository is organized to help you efficiently benchmark vehicle detection algorithms on Flash LiDAR data. Here’s how to navigate the folders and which dataset format to download for each deep learning workflow:

"YOLOX_ObjectDetection/" 
Contains code, scripts, and configurations for running the YOLOX 2D detection model.
Required data: Download the 2D Image Format dataset (range and intensity as uint16x3 images).

"SalsaNext_SemanticSegmentation/" 
Includes all necessary code and scripts for the SalsaNext segmentation model, which operates on 5-channel LiDAR images.
Required data: Download the 5-Channel Multi-Channel Image Format dataset (128x128 images with X, Y, Z, Range, Intensity channels).

"RandLANet_SemanticSegmentation/" 
Provides code and scripts for the RandLA-Net segmentation model, designed for processing point cloud (PCD) data.
Required data: Download the Point Cloud Format dataset (point cloud files with location and intensity information).

"PreProcessingScripts/" 
Offers utilities and functions for general point cloud processing tasks such as filtering and visualization. These are the scripts that we used to convert the original raw data to the other formats downloadable in the repo. If you want to make modifications on how the data was processed you can use these as a starting point. 
Required data: Download the Raw Data Format dataset (Array including the recorded range and intensity).

## License

The license for the scripts is available in the License.txt file in this GitHub repository.
The license for the dataset is included in the compressed folder you can download from the links above in a separate License.txt file. 

## Community Support
[MATLAB Central](https://www.mathworks.com/matlabcentral)

Copyright 2025 The MathWorks, Inc.

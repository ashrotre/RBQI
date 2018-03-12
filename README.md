
BUILD DEPENDENCIES:
1. CMake
2. OpenCV 3.1
3. OpenMP (Note: LLVM compiler does not support OpenMP)
4. Boost

COMPILING:

The sources are tested in Ubuntu 16.04 (64-bit), Eclipse Platform Version 3.8.1 and MacOSX 10.10, Xcode version 7.2.1

	Linux:
		1. Create build/ folder
		2. cd buid/1	`5
		3. cmake -G"Eclipse CDT4 - Unix Makefiles" ../RBQI/
		4. Open Eclipse 
		5. Import the project into the workspace (https://cmake.org/Wiki/Eclipse_CDT4_Generator)
		6. Build the project in eclipse
	MacOS:
		1. Create build/ folder
		2. cd build/
		3. cmake -G Xcode ../RBQI/
		4. Open the project in Xcode
		5. Product->edit scheme->Options->check the custom working directory box and set the path to build/
	(*Note: OpenMP is not supported in LLVM compiler and hence in Mac, the code does not use omp libraries and will be much slower than in Ubuntu.)

INPUT ARGUMENTS:

	<binary> <1-to run for entire dataset, 0-for single image> <Level(default=3)> \
              <Neighborhood (default=16)> <beta_s(default=3.5)> beta_c<default=3.5)> \
              <Reference Image Path, only for single image> <Reconstructed Image Path, only for single image>

	To run all images in the dataset:
		1. provide a csv file with the list of input images and corresponding references (examples: inputFiles.csv/ inputFiles_SBM.csv). 
		2. Output cvs to write the calculated RBQI corresponding to all images.

	To run the entire dataset:
		calcRBQI 1 3 16 3.5 3.5
	To run single reference-reconstructed image pair:
		calcRBQI 1 3 16 3.5 3.5 ../Input/Reference Background/Building.JPG ../Input/Reconstructed Background/Building/BkgEstimator.jpg

SOURCES:

main.cpp - main function. Accepts the input arguments and outputs calculated RBQI

calcDistortionMaps.cpp - calculates the distortion maps for both structure and color

ssim_CS_search.cpp - Performs the similarity search in the neighborhood of the pixel 

findBlockType.cpp - This function finds the label for every pixel in an image using the technique in “Post-Processing for artifact reduction in JPEG-compressed images.”

findAJNCD.cpp - This function calculates the Just noticeable color difference for each pixel in the input image by considering the chroma and local luminance texture as proposed in paper: "Colour image compression based on the measure of just noticeable colour difference.”

calcDR.cpp - Pools the distortion for the foveated regions.

calcSimScore.cpp - Calculated the RBQI for the given reference-reconstructed image pair at given scale

Utils.cpp/Utils.hpp - provides a support class for binary files reading and writing operations.
	
FOLDER CONTENTS:

	inputFiles.csv, inputFiles_SBM.csv - lists of all files in the ReBaQ and S-ReBaQ datasets rest.

NOTE:

To use ReBaQ / S-ReBaQ databases (Link: https://drive.google.com/drive/folders/1bg8YRPIBcxpKIF9BIPisULPBPcA5x-Bk?usp=sharing_eil&ts=5aa5e096), use the inputFiles.csv / inputFiles_SBM.csv. To use other databases, please create CSV files in the same format.

Before compiling, please update the following paths in main.cpp:
1. inputFiles.csv / inputFiles_SBM.csv
2. folder to input image sequences
3. folder to reconstructed background images
4. folder for output files

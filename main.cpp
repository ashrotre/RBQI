//
//  main.cpp
//  calcContrastStructMSIndex_Ref
//
//  Created by Aditee Shrotre on 2/1/16.
//  Copyright © 2016 Aditee Shrotre. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include "Utils.hpp"

#ifdef __APPLE__
	#include <libiomp/omp.h>
#endif


using namespace std;
using namespace cv;

extern double calcSimScore(Mat const& refDI, Mat const& imDI,
                           int nhood, float beta_s, float beta_c,
                           string dsFile, string dcFile,
                           bool useFiles);

typedef struct {
   string dataset;
   string image;
   float score;
} imgdata;

struct MatchPathSeparator
{
   bool operator()( char ch ) const
   {
      return ch == '/';
   }
};

std::string
basename( std::string const& pathname )
{
   return std::string(
                      std::find_if( pathname.rbegin(), pathname.rend(),
                                   MatchPathSeparator() ).base(),
                      pathname.end() );
}

void getImageData(string line, imgdata &data)
{
   string delim = ",";
   size_t pos = 0;
   
   while ((pos = line.find(delim)) != std::string::npos) {
      data.dataset = line.substr(0, pos);
      line.erase(0, pos + delim.length());
   }
   data.image = line.c_str();
}

void findFileToRead(string& dsFile, string& dcFile,
                    string& refDIFile, string& imDIFile,
                    string dataset, string image, int level)
{
	string dirPath = "../output/";
	dirPath = dirPath+dataset;

	// Check if the binary file exists
	// change .JPG extension to .bin for binary file
	const std::string ext(".jpg");
	if (image != ext &&
		image.size() > ext.size() &&
		image.substr(image.size() - ext.size()) == ".jpg")
	{
		// if so then strip them off
		image = image.substr(0, image.size() - ext.size());
	}
	dsFile = dirPath+"/"+image+"_dsMap_max_1minusq_" + to_string(level) +".bin";
	dcFile = dirPath+"/"+image+"_dcMap_" + to_string(level) +".bin";
   
   refDIFile = dirPath+"/"+dataset+"_DI_"+ to_string(level) +".jpg";
   imDIFile = dirPath+"/"+image+"_DI_"+ to_string(level) +".jpg";
}

double calcQI(const Mat& Ref, const Mat& Im, int nLevel, int nhood,
              float beta_s, float beta_c, string dataset=NULL, string image=NULL)
{
   Mat refDI; // stores the downsampled Image
   Ref.copyTo(refDI);
   Mat imDI; // stores the downsampled Image
   Im.copyTo(imDI);
   
   double scores[nLevel]; // Stores the scores at different level
   double quality_idx = 0.0;
   
   /* Find the Quality Indices for the number of levels. */
   for(int level = 0; level < nLevel; level++)
   {
      // set the use file to true if the files are already written and we just need to read them
	   string dsFile, dcFile, refDIFile, imDIFile;
	   if (!dataset.empty()  && !image.empty())
		   findFileToRead(dsFile, dcFile, refDIFile, imDIFile,
                        dataset, image, level);
      imwrite(refDIFile, refDI);
      imwrite(imDIFile, imDI);
      
      // Calculate the score for the current level
	   scores[level] = calcSimScore(refDI, imDI, nhood,
                                   beta_s, beta_c,
                                   dsFile, dcFile,
                                   false);
	   // Down-sample the images
	   resize(refDI, refDI, Size(), 0.5, 0.5, INTER_AREA);
	   resize(imDI, imDI, Size(), 0.5, 0.5, INTER_AREA);

	   // Accumulate the scores at individual levels
	   quality_idx += scores[level];
   }
   
   /* Calculate the final score as the log10 of the sum of scores at all levels */
   quality_idx = log10(quality_idx);
   return quality_idx;
}


int main(int argc, const char * argv[])
{
   bool runAll = 0;
   int nLevel = 3;
   int nhood = 16;
   float beta_s = 3.5;
   float beta_c = 3.5;
   string refPath, imPath;
   Mat Ref, Im;
   if (argc < 2)
   {
      cout << "Usage: <binary> <1-to run for entire dataset, 0-single image> <Level(default=3)> \
              <Neighborhood (default=16)> <beta_s(default=3.5)> <beta_c(default=3.5)> \
              <Ref Image Path, if single image> <Reconstructed Image Path, if single image> " << endl;
      exit(1);
   }
   if (argc == 2)
   {
      runAll = atoi(argv[1]);
   }
   else if (argc > 2)
   {
      runAll = atoi(argv[1]);
      if (argc == 3)
      {
         nLevel = atoi(argv[2]);
      }
      if (argc == 4)
      {
         nLevel = atoi(argv[2]);
         nhood = atoi(argv[3]);
      }
      if (argc == 5)
      {
         nLevel = atoi(argv[2]);
         nhood = atoi(argv[3]);
         beta_s = atof(argv[4]);
      }
      if (argc == 6)
      {
         nLevel = atoi(argv[2]);
         nhood = atoi(argv[3]);
         beta_s = atof(argv[4]);
         beta_c = atof(argv[5]);
      }
      if (!runAll && argc <= 8)
      {
         nLevel = atoi(argv[2]);
         nhood = atoi(argv[3]);
         beta_s = atof(argv[4]);
         beta_c = atof(argv[5]);
         refPath = argv[6];
         Ref = imread(refPath);
         if (Ref.empty())
         {
            cout << "Failed to open the Reference File:" << argv[6] << "\n Exiting.." << endl;
            exit(1);
         }
         imPath = argv[7];
         Im = imread(imPath);
         if (Im.empty())
         {
            cout << "Failed to open the Image File:" << argv[7] << "\n Exiting.." << endl;
            exit(1);
         }
      }
   }
   else if (!runAll && argc < 7)
   {
      cout << "Provide the Image files to be compared. \n";
      cout << "Usage: <binary> <1-to run for entire dataset, 0-single image> <Level(default=3)> \
               <Neighborhood (default=16)> <beta_s(default=3.5)> beta_c<default=3.5)> \
               <Ref Image Path, if single image> <Reconstructed Image Path, if single image> " << endl;
      exit(1);
   }

   /* If we need to run all the images, find the refs and corresponding Im files */
   if (runAll)
   {
      // Read the list of the files to be compared
      vector<imgdata> files;
      string inpfile("../inputFiles.csv");
      
      ifstream imglist;
      imglist.open(inpfile.c_str(), ios::in);
      if (imglist.is_open()) {
         string line;
         while (!imglist.eof()) {
            getline(imglist, line);
            if (line.length()) {
               if (*line.rbegin() == '\r')
                  line.erase(line.length() - 1);
               imgdata im;
               getImageData(line, im);
               files.push_back(im);
            }
         }
         imglist.close();
      }
      
      // Find the ref and the corresponding image.
      for (int i = 0; i < int(files.size()); i++)
      {
         imgdata im = files.at(i);
         string refPath ("../Input/ReBaQ/Reference Background/");
         refPath = refPath + im.dataset + ".JPG";
         
         string imPath ("../Input/ReBaQ/Reconstructed Background/");
         imPath = imPath + im.dataset + "/" + im.image;
         
         Ref = imread(refPath);
         Im = imread(imPath);
         files.at(i).score = calcQI(Ref, Im, nLevel, nhood, beta_s, beta_c, im.dataset, im.image);
         
         cout << im.dataset << ", " << im.image << ": " << files.at(i).score << endl;
      }
      
      // Write all results to csv file
      ofstream ofs("../output.csv", std::ofstream::out);
      ofs << "Dataset, Image, Score" << endl;
      for (int i = 0; i < int(files.size()); i++) {
         imgdata im = files.at(i);
         ofs << im.dataset << "," << im.image << "," << im.score << endl;
      }
      ofs.close();
      
   }
   /* Otherwise calculate the score for only the input images */
   else
   {
      
      // find the dataset name and remove the extension.
      string dataset = basename(refPath);
      const std::string ext(".JPG");
      if (dataset != ext &&
          dataset.size() > ext.size() &&
          dataset.substr(dataset.size() - ext.size()) == ".JPG")
      {
         // if so then strip them off
         dataset = dataset.substr(0, dataset.size() - ext.size());
      }
      string image = basename(imPath);
      
      float quality_idx = calcQI(Ref, Im, nLevel, nhood, beta_s, beta_c, dataset, image);
      cout << "Quality Index of the image: " << quality_idx << endl;
   }
   
   return 0;
}


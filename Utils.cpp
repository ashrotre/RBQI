//
//  Utils.cpp
//  NR_QualityIdx
//
//  Created by Aditee Shrotre on 3/25/16.
//  Copyright © 2016 Aditee Shrotre. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <boost/filesystem.hpp>
#include "Utils.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

extern vector<int> getLocalMaximum(InputArray _src, int smooth_size = 9, int neighbor_size = 3, float peak_per = 0.5);

bool Utils::directoryExists( const std::string &directory )
{
   if( !directory.empty() )
   {
      if( access(directory.c_str(), 0) == 0 )
      {
         struct stat status;
         stat( directory.c_str(), &status );
         if( status.st_mode & S_IFDIR )
            return true;
      }
   }
   // if any condition fails
   return false;
}

bool Utils::createDirectory( const std::string &dirPath )
{
   boost::filesystem::path dir(dirPath);
   if ( boost::filesystem::create_directories(dir) )
      return true;
   
   return false;
}

bool Utils::fileExists( const std::string &filename )
{
   if( !filename.empty() )
   {
      if( access(filename.c_str(), 0) == 0 )
      {
         struct stat status;
         stat( filename.c_str(), &status );
         if( !(status.st_mode & S_IFDIR) )
            return true;
      }
   }
   // if any condition fails
   return false;
}

bool Utils::imageWrite(const std::string &filename, const Mat &data, int channels, size_t size)
{
   bool result = false;
   if (!filename.empty()) {
      ofstream of;
      of.open(filename, ios::out | ios::binary);
      of.write((char *) data.data, data.rows * data.cols * channels * size);
      of.close();
      result = true;
   }
   return result;
}

void Utils::imageRead(const std::string &filename, Mat &data, int channels, size_t size)
{
   if (!filename.empty()) {
      ifstream ifs(filename, ios::binary);
      if (ifs.is_open()) {
         ifs.read((char *) data.data, data.rows * data.cols * channels * size);
      }
      ifs.close();
   }
}

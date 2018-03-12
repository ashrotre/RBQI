//
//  Utils.hpp
//  NR_QualityIdx
//
//  Created by Aditee Shrotre on 3/25/16.
//  Copyright © 2016 Aditee Shrotre. All rights reserved.
//

#ifndef Utils_hpp
#define Utils_hpp

#include <iostream>
#include <opencv2/core.hpp>

using namespace cv;

class Utils {
public:
   bool directoryExists(const std::string &directory);
   bool createDirectory(const std::string &dirPath);
   bool fileExists(const std::string &filename);
   bool imageWrite(const std::string &filename, const Mat &data, int channels, size_t size);
   void imageRead(const std::string &filename, Mat &data, int channels, size_t size);
   void calcGrayHistogram(cv::Mat& img, cv::MatND& hist, bool plot);
   void plotHistogram(MatND &hist, int bins);
//   std::vector<int> getLocalMaximum(InputArray _src, int smooth_size, int neighbor_size, float peak_per);
};

#endif /* Utils_hpp */

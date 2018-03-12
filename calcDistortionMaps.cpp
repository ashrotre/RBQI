//
//  calcQnDC.cpp
//  calcContrastStructMSIndex_Ref
//
//  Created by Aditee Shrotre on 2/2/16.
//  Copyright © 2016 Aditee Shrotre. All rights reserved.
//

#include <iostream>
#include <time.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "Utils.hpp"

using namespace std;
using namespace cv;

extern Mat ssim_CS_search(const Mat& ref, const Mat& im, float K,
                          int gaussian_window, float gaussian_sigma, float L,
                          int nhood);

void calcDistortionMaps(Mat& dsMap, Mat& dcMap,
                        Mat const& ref, Mat const& im,
                        int nhood)
{
   /* Calculate the Quality Map using only the SSIM - contrast and structure
    * components */
   // Convert to gray and use only the single channel for the SSIM calculation
   Mat refG, imG;
   cvtColor(ref, refG, CV_BGR2GRAY);
   cvtColor(im, imG, CV_BGR2GRAY);
   
   float K = 0.03;
   int gaussian_window = 11;
   int gaussian_sigma = 2.0;
   float L = 255.0;
   time_t start, end;
   time(&start);
   Mat qMap = ssim_CS_search(refG, imG, K, gaussian_window, gaussian_sigma, L, nhood);
   time(&end);
   double diff = difftime(end, start);
   cout << "Elapsed time is :  " << diff <<" seconds." << endl;
   
   // qMap has values between -1<=qMap <=1
   // Convert the similarity map to structural dissim map, dmap = (1-qMap)/2 0<=dsMap<=1
   // higher value of dsMap represents more dissimilarity between the 2 images
   dsMap = (1-qMap);
   dsMap.convertTo(dsMap, CV_32F);
   
   // Uncomment the below lines to write the quality map images
   Mat dsMap_t;
   dsMap.convertTo(dsMap_t, CV_8U, 255);
   dsMap_t=255-dsMap_t; // invert for display purposes
   namedWindow("QualityMap");
   imshow("QualityMap", dsMap_t);
   waitKey();
   imwrite("../output/dsMap_escalator.jpg",dsMap_t);
   
   /* Calculate the eucledian distance in the Lab color space */
   // Convert to Lab color space and scale to the correct values
   Mat refLab, imLab;
   vector<Mat> ch(3);
   cvtColor(ref, refLab, CV_BGR2Lab);
   refLab.convertTo(refLab, CV_32F);
   split(refLab, ch);
   ch[0] = ch[0] * 100.0/255.0;
   ch[1] = ch[1] - 128.0;
   ch[2] = ch[2] - 128.0;
   merge(ch, refLab);
   cvtColor(im, imLab, CV_BGR2Lab);
   imLab.convertTo(imLab, CV_32F);
   split(imLab,ch);
   ch[0] = ch[0] * 100.0/255.0;
   ch[1] = ch[1] - 128.0;
   ch[2] = ch[2] - 128.0;
   merge(ch, imLab);

   // Smooth the images
   GaussianBlur(refLab, refLab, Size(gaussian_window, gaussian_window), 2.0);
   GaussianBlur(imLab, imLab, Size(gaussian_window, gaussian_window), 2.0);

   Mat diffLab = (refLab - imLab);
   pow(diffLab, 2, dcMap);
   transform(dcMap, dcMap, Matx13f(1,1,1));
   sqrt(dcMap, dcMap);

}

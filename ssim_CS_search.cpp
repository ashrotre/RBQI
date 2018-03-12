//
//  ssim_CS.cpp
//  calcContrastStructMSIndex_Ref
//
//  Created by Aditee Shrotre on 2/2/16.
//  Copyright © 2016 Aditee Shrotre. All rights reserved.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


double ssim_idx(const Mat& ref_block, const Mat& im_block,
               double ref_mu, double ref_sig_sq,
               const Mat& mu2_rect, const Mat& sigma2_sq_rect,
               int nhood, float C, int gaussian_window, float gaussian_sigma)
{
   // double ssim_val;
   int bd = floor(gaussian_window/2);
   
   int block_w = im_block.cols - 2*bd;
   int block_h = im_block.rows - 2*bd;

   Mat kernel = getGaussianKernel(gaussian_window, gaussian_sigma);
   kernel = kernel * kernel.t();

   Mat map = Mat::zeros(block_h, block_w, CV_64F);

   for (int i = bd; i < block_h+bd; i++)
   {
      for (int j = bd; j < block_w+bd; j++)
      {
         double mu1_mu2 = ref_mu * mu2_rect.at<double>(i-bd, j-bd);
         double sigma1_sq = ref_sig_sq;
         double sigma2_sq = sigma2_sq_rect.at<double>(i-bd, j-bd);
         
         Mat im_curr_block(im_block(Rect(j-bd, i-bd, gaussian_window, gaussian_window)));
         Mat ref_im_mul = ref_block.mul(im_curr_block);
         double sigma12 = sum(ref_im_mul.mul(kernel))[0] - mu1_mu2; // manually convolve
         
         if (C <= 0)
            cout << "Error: C is not greater than 0." << endl;
         
         double numerator = 2*sigma12 + C;
         double denominator = sigma1_sq + sigma2_sq + C;
         
         map.at<double>(i-bd, j-bd) = numerator/denominator;
      }
   }

   double idx;
    minMaxLoc(map, NULL, &idx); // For max similarity
   return idx;
}

Mat ssim_CS_search(const Mat& refG, const Mat& imG, float K,
                   int gaussian_window, float gaussian_sigma, float L, int nhood)
{
   int H = refG.rows;
   int W = refG.cols;
   
   Mat ref, im;
   refG.convertTo(ref, CV_64F);
   imG.convertTo(im, CV_64F);
   
   Mat mu1, mu2, mu1_mu2, mu1_sq, mu2_sq, sigma1_sq, sigma2_sq, sigma12;
   
   float C = (K*L)*(K*L);
   
   // Calculating mu and mu square, mu1_mu2
   GaussianBlur(ref, mu1, Size(gaussian_window, gaussian_window), gaussian_sigma);
   GaussianBlur(im, mu2,  Size(gaussian_window, gaussian_window), gaussian_sigma);
   
   pow(mu1, 2, mu1_sq);
   pow(mu2, 2, mu2_sq);
   
   // Calculating sigma, sigma_sq
   Mat ref_sq, im_sq, ref_im;
   pow(ref, 2, ref_sq);
   pow(im, 2, im_sq);
   GaussianBlur(ref_sq, sigma1_sq, Size(gaussian_window, gaussian_window), gaussian_sigma);
   sigma1_sq = sigma1_sq - mu1_sq;
   GaussianBlur(im_sq, sigma2_sq, Size(gaussian_window, gaussian_window), gaussian_sigma);
   sigma2_sq = sigma2_sq - mu2_sq;

   int bd = floor(gaussian_window/2);

   Mat im_border, ref_border;
   copyMakeBorder(ref, ref_border, bd, bd, bd, bd, BORDER_DEFAULT);    // replicate the image at borders
   copyMakeBorder(im, im_border, bd, bd, bd, bd, BORDER_DEFAULT);    // replicate the image at borders
   
   Mat qMap = Mat::zeros(ref.rows, ref.cols, CV_64F); // stores the Quality Map
   
#pragma omp parallel for
   for (int h = bd; h < H+bd; h++)
   {
      for (int w = bd; w < W+bd; w++)
      {
         double ref_mu = mu1.at<double>(h-bd,w-bd);
         double ref_sig_sq = sigma1_sq.at<double>(h-bd, w-bd);
         
         // Extract the stats of the image neighborhood block
         int h_start = (h-bd) - nhood/2;
         h_start = (h_start < 0) ? 0 : h_start;
         int h_end = (h-bd) + nhood/2;
         h_end = (h_end >= H) ? H-1 : h_end;
         
         int w_start = (w-bd) - nhood/2;
         w_start = (w_start < 0) ? 0 : w_start;
         int w_end = (w-bd) + nhood/2;
         w_end = (w_end >= W) ? W-1 : w_end;
         
         Mat mu2_rect(mu2(Rect(w_start, h_start, w_end-w_start, h_end-h_start)));
         Mat sigma2_sq_rect(sigma2_sq(Rect(w_start, h_start, w_end-w_start, h_end-h_start)));
         
         // Extract the image neighborhood block
         h_start = h - nhood/2 - bd;
         h_start = (h_start < 0) ? 0 : h_start;
         h_end = h + nhood/2 + bd;
         h_end = (h_end >= H+2*bd) ? H+2*bd-1 : h_end;
         
         w_start = w - nhood/2 - bd;
         w_start = (w_start < 0) ? 0 : w_start;
         w_end = w + nhood/2 + bd;
         w_end = (w_end >= W+2*bd) ? W+2*bd-1 : w_end;
         
         Mat im_block(im_border(Rect(w_start, h_start, w_end-w_start, h_end-h_start)));
         
         // Extract the Reference and the image block
         Mat ref_block(ref_border(Rect(w-bd, h-bd, gaussian_window, gaussian_window)));
//         Mat im_block(im_border(Rect(w, h, gaussian_window, gaussian_window)));
         
         qMap.at<double>(h-bd, w-bd) = ssim_idx(ref_block, im_block,
                                               ref_mu, ref_sig_sq,
                                               mu2_rect, sigma2_sq_rect,
                                               nhood, C, gaussian_window, gaussian_sigma);
         
      }
   }
   
   return qMap;
}

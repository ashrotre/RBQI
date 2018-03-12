#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

/* This function finds the label for every pixel in an image using the technique in
* "Post-Processing for artifact reduction in JPEG-compressed images.
* Input Args:
*   im - image to be labeled
*   blockSz - neighborhood block size to be used
*   foveatedR - foveated region blocks used for classification
* Output Args:
*  alpha_s - alpha values as based on the label per pixel
*            1.0 - nontextured, 1000.0 - textured
*/
void findBlockType(Mat& alpha_s, Mat const& refDI, Size const& blockSz, Size const& foveatedR)
{
   // convert to gray scale
   Mat refG;
   cvtColor(refDI, refG, CV_BGR2GRAY);
   
   // calculate the variance at every pixel using its neighborhood
   Mat h = Mat::ones(blockSz.height, blockSz.width, CV_32F);
   float n = sum(h)[0];
   float n1 = n - 1.0;
   Mat refG_sq, varIm;
   refG.convertTo(refG, CV_32F);

   // var = (sum(I^2) - (sum(I))^2/N)/N
   Mat c1,c2;
   pow(refG, 2, refG_sq);
   filter2D(refG_sq, c1, -1, h);
   filter2D(refG, c2, -1, h);
   pow(c2, 2, c2);
   c2 = c2/n;
   varIm = max((c1-c2)/n, 0.0);

   // classify every pixel
   Mat lbl = Mat::zeros(refG.rows, refG.cols, CV_8U);
   lbl = (varIm <= 50.0)/255;
   lbl += ((varIm > 50.0)/255 & (varIm <= 1200)/255) * 2;
   lbl += ((varIm > 1200.0)/255) * 3;

   // set default to 0.8
   alpha_s = alpha_s * 0.8;
   // Classify every block of the size of foveated region into textured and non-textured
   for (int h = 0; h < refG.rows; h += foveatedR.height)
   {
      int end_h = min(h+foveatedR.height, refG.rows-1);
      for (int w = 0; w < refG.cols; w += foveatedR.width)
      {
    	  int end_w = min(w+foveatedR.width, refG.cols-1);
    	  // find the roi (block of size foveatedR)
    	  Rect rect = Rect(w, h, end_w-w, end_h-h);
    	  Mat roi_lbl = Mat(lbl, rect);
    	  Mat roi_alpha = Mat(alpha_s, rect);

    	  // for every block find the number of pixels of type - uniform/texture/edge
    	  int nUniform = countNonZero(roi_lbl == 1);
    	  int nEdge = countNonZero(roi_lbl == 3);
    	  float pUniform = ((float)foveatedR.area()-(float)nEdge)/(float)foveatedR.area();
    	  if (nUniform >= 50 && nUniform <= 64 && nEdge == 0)
    		  roi_alpha.setTo(0.8);
    	  else if (nUniform >= 20 && nUniform <= 49 && nEdge == 0)
    		  roi_alpha.setTo(100.0);
    	  else if (nUniform >= 0 && nUniform <= 19 && nEdge == 0)
    		  roi_alpha.setTo(100.0);
    	  else if (pUniform < 0.65 && nEdge > 0 && nEdge <= 19)
    		  roi_alpha.setTo(1.0);
    	  else
    		  roi_alpha.setTo(0.8);
      }
   }
}

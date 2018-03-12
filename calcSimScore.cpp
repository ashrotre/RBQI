#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Utils.hpp"

using namespace std;
using namespace cv;

extern void calcDistortionMaps(Mat& dsMap, Mat& dcMap,
                               Mat const& refDI, Mat const& imDI,
                               int nhood);
extern void findBlockType(Mat& alpha_s, Mat const& refDI, Size const& blockSz, Size const& foveatedR);
extern void findAJNCD(Mat& alpha_c, Mat const& refDI, Size const& foveatedR);
extern void calcDR(Mat& DR, Mat const& dMap, Mat const& alpha, float beta, Size const& foveatedR);


double calcSimScore(Mat const& refDI, Mat const& imDI,
                    int nhood, float beta_s, float beta_c,
                    string dsFile, string dcFile,
                    bool useFile=true)
{
	Utils sys;
	Mat dsMap, dcMap;

	// If we need to use the file but no file is provided then calculate QnDC and
	// then write the file to the location
	if (useFile == false)
	{
		calcDistortionMaps(dsMap, dcMap, refDI, imDI, nhood);
		// write the file
		if(!dsFile.empty() && !dcFile.empty())
		{
			sys.imageWrite(dsFile, dsMap, 1, 4);
			sys.imageWrite(dcFile, dcMap, 1, 4);
		}
	}
	else
	{
		if (sys.fileExists(dsFile) && sys.fileExists(dcFile))
		{
         dsMap = Mat::zeros(refDI.rows, refDI.cols, CV_32FC1);
         sys.imageRead(dsFile, dsMap, 1, 4);
         
         dcMap = Mat::zeros(refDI.rows, refDI.cols, CV_32FC1);
         sys.imageRead(dcFile, dcMap, 1, 4);
		}
		else
		{
			cout << "Could not find the map files: " << dsFile << " & " << dcFile << endl;
			return (-1);
		}
	}
   
   /* Find the alpha_s, by classifying the blocks in to texture/non-teture */
   Size blockSz(3,3); // size of neighborhood used for calculating the variance at a pixel
   Size foveatedR(8,8); // foveated region size used for pooling the scores
   Mat alpha_s = Mat(refDI.rows, refDI.cols, CV_32FC1);
   findBlockType(alpha_s, refDI, blockSz, foveatedR);

   /* Find the AJNCD using the neighborhood statistics in the image. */
   Mat alpha_c = Mat(refDI.rows, refDI.cols, CV_32FC1);
   findAJNCD(alpha_c, refDI, foveatedR);

   /* Pool the values in the foveated region using probability model */
   int rows = ceil(refDI.rows/foveatedR.height)-2; // ignore the edge blocks
   int cols = ceil(refDI.cols/foveatedR.width)-2;
   Mat DR_s = Mat(rows, cols, CV_32F);
   Mat DR_c = Mat(rows, cols, CV_32F);
   calcDR(DR_s, dsMap, alpha_s, beta_s, foveatedR);
   patchNaNs(DR_s,0.0);
   calcDR(DR_c, dcMap, alpha_c, beta_c, foveatedR);
   patchNaNs(DR_c,0.0);

   Mat DR = DR_s + DR_c;

   double score = sum(DR)[0];
   return score;
}

						  

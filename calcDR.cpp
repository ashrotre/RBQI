#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

/* This function calculcates the distortions in the foveated region
 * using the psychometric model.
 */
void calcDR(Mat& DR, Mat const& dMap, Mat const& alpha, float beta, Size const& foveatedR)
{
	// Form image grid of non-overlapping block locations, ignore the edge blocks
	for (int h = foveatedR.height; h < dMap.rows-foveatedR.height; h += foveatedR.height)
	{
	  int end_h = min(h+foveatedR.height, dMap.rows-(2*foveatedR.height));
	  for (int w = foveatedR.width; w < dMap.cols-foveatedR.width; w += foveatedR.width)
	  {
		  int end_w = min(w+foveatedR.width, dMap.cols-(2*foveatedR.width));
		  // find the roi (block of size foveatedR)
		  Rect rect = Rect(w, h, end_w-w, end_h-h);
		  Mat roi_dmap = Mat(dMap, rect);
		  Mat roi_alpha = Mat(alpha, rect);

		  Mat tmp;
		  pow((roi_dmap/roi_alpha), beta, tmp);
		  DR.at<float>(h/foveatedR.height-1, w/foveatedR.width-1) = sum(tmp)[0];

	  }
	}
}

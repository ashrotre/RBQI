#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Utils.hpp"

using namespace std;
using namespace cv;

/* This function calculates the Just noticeable color difference for each
 * pixel in the input image by considering the chroma and local luminance
 * texture as proposed in paper:
 * "Colour image compression based on the measure of just noticeable
 *  colour difference"
 * Input Args:
 *   refID - image for which JNCD has to be calculated
 *   foveatedR - foveated region blocks used for classification
 * Output Args:
 *  alpha_c - JNCD values at every pixel
 */
void findAJNCD(Mat& alpha_c, Mat const& refDI, Size const& foveatedR)
{
	Utils sys;

	// Change the color space to Lab
	Mat refLab;
	cvtColor(refDI, refLab, CV_BGR2Lab);
	refLab.convertTo(refLab, CV_32F);
	double m, M;
	minMaxLoc(refLab, &m, &M);
	float JNCD_Lab = 2.3;

	vector<Mat> lab_ch(3);
	split(refLab, lab_ch);
	lab_ch[0] = lab_ch[0] * 100.0/255.0;
	lab_ch[1] = lab_ch[1] - 128.0;
	lab_ch[2] = lab_ch[2] - 128.0;

	Mat c;
	magnitude(lab_ch[1], lab_ch[2], c);
	Mat sc = 1.0 + 0.045*c;

	// Calcalate EL
	Mat h = Mat::ones(foveatedR.height, foveatedR.width, CV_32F);
	float n = sum(h)[0];
	Mat EL;
	filter2D(lab_ch[0], EL, -1, h/n);
	Mat rhoEL;
	Mat tmp1 = Mat::zeros(EL.rows, EL.cols, CV_8U);
	Mat tmp2 = Mat::zeros(EL.rows, EL.cols, CV_8U);
	Mat tmp3 = Mat::zeros(EL.rows, EL.cols, CV_8U);
	Mat tmp4 = Mat::zeros(EL.rows, EL.cols, CV_8U);

	tmp1 = (EL <= 20.0)/255;
	tmp1.convertTo(tmp1, CV_32F, 0.09);
	tmp2 += ((EL > 20.0)/255 & (EL <= 40.0)/255);
	tmp2.convertTo(tmp2, CV_32F, 0.07);
	tmp3 += ((EL > 40.0)/255 & (EL <= 60.0)/255);
	tmp3.convertTo(tmp3, CV_32F, 0.05);
	tmp4 += ((EL > 60.0)/255.0);
	tmp4.convertTo(tmp4, CV_32F, 0.08);
	rhoEL = tmp1 + tmp2 + tmp3 + tmp4;

	// Calculate the image gradients
	Mat dx, dy, L;
	Sobel(lab_ch[0], dx, -1, 1, 0, 3);
	Sobel(lab_ch[0], dy, -1, 0, 1, 3);
	magnitude(dx,dy,L);
	h = Mat::ones(foveatedR, CV_32F);
	Mat dL;
	dilate(L, dL, h);

	Mat sl = rhoEL.mul(dL) + 1.0;
	alpha_c = JNCD_Lab * (sl.mul(sc));

}

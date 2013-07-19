#include "C:\OpenCV2.3\include\opencv2\opencv.hpp"
#include "CreateDegradingMatrix.h"
#include "MultiplySparseMatrix.h"

cv::SparseMat CreateDegradingMatrixAndDegradedImage(cv::Mat& src,cv::Mat& dst,cv::Point2d move,int mfactor)
{
	cv::SparseMat A=CreateDegradingMatrix(src,mfactor,move);

	cv::Mat svec;
	src.reshape(3,src.cols*src.rows).convertTo(svec,CV_32FC3);
	cv::Mat dvec(src.cols*src.rows/(mfactor*mfactor),1,CV_32FC3);

	MultiplySparseMatrix(A,svec,dvec,false);

	dvec.reshape(3,dst.rows).convertTo(dst,CV_8UC3);

	return A;

}
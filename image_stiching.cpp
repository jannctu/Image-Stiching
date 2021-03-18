#include "opencv2\opencv.hpp"
#include <vector>
#include <stdio.h>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <cstdlib>
#include <time.h>

//using namespace std;
using namespace cv;
void PrintMatrix(CvMat *Matrix, int Rows, int Cols);
char * stichingImage(char* str1, char* str2);
char * strToCharStar(string inp);

//
int main(){
	clock_t tStart = clock();
	/*  Automatic stitching images
	* image lists store in vector. 
	* if something goes wrong please check using manual input. (Line:56)
	*/
	vector<string> inputs; // input images
	inputs.push_back("A001");inputs.push_back("A002");inputs.push_back("A003");
	inputs.push_back("A004");inputs.push_back("A005");inputs.push_back("A006");
	inputs.push_back("A007");inputs.push_back("A008");inputs.push_back("A009");
	
	int cnt = inputs.size();
	vector<string> checked;
	char* result;
	for (int i = 0; i < cnt-1; i++){
		std::vector<std::string>::const_iterator it = std::find(checked.begin(), checked.end(), inputs.at(i));
		if (it == checked.end()){
			result = strToCharStar(inputs.at(i));
			for (int j = i + 1; j < cnt; j++){
				if (strcmp(result, "Failed!") != 0){
					std::cout << "============================= " << std::endl;
					std::cout << result << " + " << inputs.at(j) << std::endl;
					result = stichingImage(result, strToCharStar(inputs.at(j)));
					//result = "No Matching Keypoints";
					std::cout << " result : " << result << std::endl;
					std::cout << "Time : " << (double)(clock() - tStart) / CLOCKS_PER_SEC << std::endl;
					std::cout << "============================= " << std::endl;

					if (strcmp(result, "Failed!") != 0){
						checked.push_back(inputs.at(j));
					}
				}
			}
		}
	}


	/* 
	* if something goes wrong please check using manual input.
	* params are image name (jpg ext).
	* params are just file name, no need ext.
	* like the following example : 
	stichingImage("A001", "A002");
	*/
	return 0;
}

char * strToCharStar(string inp){
	
	char *cstr = new char[inp.length() + 1];
	strcpy(cstr, inp.c_str());
	return cstr;
}

char * stichingImage(char* str1, char* str2){
	
	char * result = (char *)malloc(1 + strlen(str1) + 1 + strlen(str1) + 4);
	// clock_t tStart = clock();
	//source image
	// image read
	char * img1_file = (char *)malloc(1 + strlen(str1) + 4);
	strcpy(img1_file, str1);
	strcat(img1_file, ".jpg");

	char * img2_file = (char *)malloc(1 + strlen(str2) + 4);
	strcpy(img2_file, str2);
	strcat(img2_file, ".jpg");
	Mat tmp = cv::imread(img1_file, 1);
	Mat in = cv::imread(img2_file, 1);

	/* threshold      = 0.04;
	edge_threshold = 10.0;
	magnification  = 3.0;    */
	std::cout << " SIFT Algorithm " << std::endl;
	// SIFT feature detector and feature extractor
	cv::SiftFeatureDetector detector(0.05, 5.0);
	cv::SiftDescriptorExtractor extractor(3.0);

	// Feature detection
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	detector.detect(tmp, keypoints1);
	detector.detect(in, keypoints2);

	// Feature display
	Mat feat1, feat2;
	cv::drawKeypoints(tmp, keypoints1, feat1, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::drawKeypoints(in, keypoints2, feat2, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imwrite("feat1.bmp", feat1);
	cv::imwrite("feat2.bmp", feat2);
	int key1 = keypoints1.size();
	int key2 = keypoints2.size();
	//printf("Keypoint1=%d \nKeypoint2=%d \n", key1, key2);

	// Feature descriptor computation
	Mat descriptor1, descriptor2;
	extractor.compute(tmp, keypoints1, descriptor1);
	extractor.compute(in, keypoints2, descriptor2);

	//printf("Descriptor1=(%d,%d) \nDescriptor2=(%d,%d)", descriptor1.size().height, descriptor1.size().width, descriptor2.size().height, descriptor2.size().width);
	//cout << endl;

	//cout << "SIFT time : " << (double)(clock() - tStart) / CLOCKS_PER_SEC << endl;
	int p1, p2, p3, p4, in_num;
	in_num = -1;
	CvMat* warp = cvCreateMat(3, 3, CV_32FC1);
	CvSize Size1;
	Size1 = cvSize((tmp.cols)*1.5, (tmp.rows)*1.5);
	IplImage *Image1 = cvCreateImageHeader(Size1, IPL_DEPTH_8U, 3);
	cvCreateData(Image1);
	cvSetZero(Image1);
	IplImage *Image2 = cvCreateImageHeader(Size1, IPL_DEPTH_8U, 3);
	cvCreateData(Image2);
	cvSetZero(Image2);
	IplImage *Image3 = cvCreateImageHeader(Size1, IPL_DEPTH_8U, 3);
	cvCreateData(Image3);
	cvSetZero(Image3);
	IplImage * img1 = cvLoadImage(img1_file);
	IplImage * img2 = cvLoadImage(img2_file);
	CvPoint2D32f srcTri[4], dstTri[4];
	CvMat* warp_mat = cvCreateMat(3, 3, CV_32FC1);
	CvMat* warp_mat_N = cvCreateMat(3, 3, CV_32FC1);
	CvMat *TransposeMatrix = cvCreateMat(8, 8, CV_32FC1);
	CvMat *Matrix2 = cvCreateMat(8, 1, CV_32FC1);
	CvMat *Matrix1 = cvCreateMat(8, 8, CV_32FC1);
	CvMat *ResultMatrix = cvCreateMat(8, 8, CV_32FC1);
	CvMat *InverseMatrix = cvCreateMat(8, 8, CV_32FC1);
	CvMat *FinalMatrix = cvCreateMat(8, 1, CV_32FC1);
	float X, Y, T, dis;
	int height, width;
	CvScalar Scalar1, Scalar2;
	height = cvGetDimSize(Image1, 0);
	width = cvGetDimSize(Image1, 1);

	IplImage * original_img = cvLoadImage(img2_file);
	int height1, width1;
	height1 = cvGetDimSize(original_img, 0);
	width1 = cvGetDimSize(original_img, 1);

	int height2, width2;
	height2 = cvGetDimSize(img1, 0);
	width2 = cvGetDimSize(img1, 1);

	// matching
	std::cout << " Matching Keypoints " << std::endl;
	double min, sum, min1, min2, min3, min4, v;
	vector<int> ivector(key1);
	vector<int> valid(key1, 0);
	for (int i = 0; i<key1; ++i){
		min = 1.79769e+308;
		for (int j = 0; j<key2; ++j){
			sum = 0;
			for (int k = 0; k < 128; ++k){
				v = descriptor1.at<float>(i, k) - descriptor2.at<float>(j, k);
				sum += v*v;
			}
			if (sum<min){
				min = sum;
				ivector[i] = j;
				if (min<128 * 16)
					valid[i] = 1;
			}
		}
	}
	int count = 0;
	for (int i = 0; i<key1; ++i)
	if (valid[i] == 1)
		count++;

	//cout << endl << endl << count << endl << endl;
	std::cout << "Match Keypoints : " <<count << std::endl;
	//cout << "matching time : " << (double)(clock() - tStart) / CLOCKS_PER_SEC << endl;

	int minimum;
	if (key1 >= key2)
		minimum = key2;
	else
		minimum = key1;

	float min_distance = (height1*height1 + width1*width1) / 16;
	/*RANSAC
		It should have modes that increse gradually
		for different type of images. depent on amount of keypoint
	*/
	int minimum_valid_key_match = 11;
	if (count > minimum_valid_key_match){
		std::cout << " RANSAC " << std::endl;
		int RANSAC_counter = 0; 
		int RANSAC_limit = count / 2;
		int RANSAC_limit_points = count * 3 / 10; 
		double RANSAC_limit_pixel = 0.3;
		//int RANSAC_limit = 1;
		//bool RANSAC_flag;
		while (in_num<RANSAC_limit_points /*|| RANSAC_counter < RANSAC_limit*/){
			p1 = rand() % key1;
			while (valid[p1] != 1)
				p1 = rand() % key1;
			p2 = rand() % key1;
			while (p2 == p1 || valid[p2] != 1)
				p2 = rand() % key1;
			p3 = rand() % key1;
			while ((p3 == p2 || p3 == p1) || valid[p3] != 1)
				p3 = rand() % key1;
			p4 = rand() % key1;
			while ((p4 == p3 || p4 == p2 || p4 == p1) || valid[p4] != 1)
				p4 = rand() % key1;

			min = 1.79769e+308;
			for (int j = 0; j<key2; ++j){
				sum = 0;
				for (int k = 0; k<128; ++k)
					sum += ((descriptor1.at<float>(p1, k) - descriptor2.at<float>(j, k))*(descriptor1.at<float>(p1, k) - descriptor2.at<float>(j, k)));
				if (sum<min){
					min = sum;
					min1 = j;
				}
			}

			min = 1.79769e+308;
			for (int j = 0; j<key2; ++j){
				sum = 0;
				for (int k = 0; k<128; ++k)
					sum += ((descriptor1.at<float>(p2, k) - descriptor2.at<float>(j, k))*(descriptor1.at<float>(p2, k) - descriptor2.at<float>(j, k)));
				if (sum<min){
					min = sum;
					min2 = j;
				}
			}
			//cout<<endl<<min<<endl;
			//cout<<min2<<endl;

			min = 1.79769e+308;
			for (int j = 0; j<key2; ++j){
				sum = 0;
				for (int k = 0; k<128; ++k)
					sum += ((descriptor1.at<float>(p3, k) - descriptor2.at<float>(j, k))*(descriptor1.at<float>(p3, k) - descriptor2.at<float>(j, k)));
				if (sum<min){
					min = sum;
					min3 = j;
				}
			}
			//cout<<endl<<min<<endl;
			//cout<<min3<<endl;

			min = 1.79769e+308;
			for (int j = 0; j<key2; ++j){
				sum = 0;
				for (int k = 0; k<128; ++k)
					sum += ((descriptor1.at<float>(p4, k) - descriptor2.at<float>(j, k))*(descriptor1.at<float>(p4, k) - descriptor2.at<float>(j, k)));
				if (sum<min){
					min = sum;
					min4 = j;
				}
			}

			// begin calculate homography
			srcTri[0].x = 0;
			srcTri[0].y = 0;
			srcTri[1].x = tmp.cols - 1;
			srcTri[1].y = 0;
			srcTri[2].x = 0;
			srcTri[2].y = tmp.rows - 1;
			srcTri[3].x = tmp.cols - 1;
			srcTri[3].y = tmp.rows - 1;

			dstTri[0].x = srcTri[0].x + tmp.cols*0.25;
			dstTri[0].y = srcTri[0].y + tmp.rows*0.25;
			dstTri[1].x = srcTri[1].x + tmp.cols*0.25;
			dstTri[1].y = srcTri[1].y + tmp.rows*0.25;
			dstTri[2].x = srcTri[2].x + tmp.cols*0.25;
			dstTri[2].y = srcTri[2].y + tmp.rows*0.25;
			dstTri[3].x = srcTri[3].x + tmp.cols*0.25;
			dstTri[3].y = srcTri[3].y + tmp.rows*0.25;

			//cvGetPerspectiveTransform( srcTri, dstTri, warp_mat );
			float Array1[] = { srcTri[0].x, 0, srcTri[1].x, 0, srcTri[2].x, 0, srcTri[3].x, 0, srcTri[0].y, 0, srcTri[1].y, 0, srcTri[2].y, 0, srcTri[3].y, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, srcTri[0].x, 0, srcTri[1].x, 0, srcTri[2].x, 0, srcTri[3].x, 0, srcTri[0].y, 0, srcTri[1].y, 0, srcTri[2].y, 0, srcTri[3].y, 0, 1, 0, 1, 0, 1, 0, 1, -dstTri[0].x*srcTri[0].x, -dstTri[0].y*srcTri[0].x, -dstTri[1].x*srcTri[1].x, -dstTri[1].y*srcTri[1].x, -dstTri[2].x*srcTri[2].x, -dstTri[2].y*srcTri[2].x, -dstTri[3].x*srcTri[3].x, -dstTri[3].y*srcTri[3].x, -dstTri[0].x*srcTri[0].y, -dstTri[0].y*srcTri[0].y, -dstTri[1].x*srcTri[1].y, -dstTri[1].y*srcTri[1].y, -dstTri[2].x*srcTri[2].y, -dstTri[2].y*srcTri[2].y, -dstTri[3].x*srcTri[3].y, -dstTri[3].y*srcTri[3].y };

			cvSetData(TransposeMatrix, Array1, TransposeMatrix->step);
			float Array2[] = { dstTri[0].x, dstTri[0].y, dstTri[1].x, dstTri[1].y, dstTri[2].x, dstTri[2].y, dstTri[3].x, dstTri[3].y };

			cvSetData(Matrix2, Array2, Matrix2->step);
			cvTranspose(TransposeMatrix, Matrix1);
			/*cvMatMul(TransposeMatrix,Matrix1,ResultMatrix);
			cvInvert(ResultMatrix,InverseMatrix,CV_SVD);
			cvMatMul(InverseMatrix,TransposeMatrix,ResultMatrix);*/
			cvInvert(Matrix1, ResultMatrix, CV_SVD);//
			cvMatMul(ResultMatrix, Matrix2, FinalMatrix);
			cvmSet(warp_mat, 0, 0, cvmGet(FinalMatrix, 0, 0));
			cvmSet(warp_mat, 0, 1, cvmGet(FinalMatrix, 1, 0));
			cvmSet(warp_mat, 0, 2, cvmGet(FinalMatrix, 2, 0));
			cvmSet(warp_mat, 1, 0, cvmGet(FinalMatrix, 3, 0));
			cvmSet(warp_mat, 1, 1, cvmGet(FinalMatrix, 4, 0));
			cvmSet(warp_mat, 1, 2, cvmGet(FinalMatrix, 5, 0));
			cvmSet(warp_mat, 2, 0, cvmGet(FinalMatrix, 6, 0));
			cvmSet(warp_mat, 2, 1, cvmGet(FinalMatrix, 7, 0));
			cvmSet(warp_mat, 2, 2, 1);
			cvInvert(warp_mat, warp_mat_N, CV_SVD);

			IplImage * src1 = cvLoadImage(img1_file);
			//cvWarpPerspective( src1, Image1, warp_mat );
			for (int i = 0; i<height; i++)
			{
				for (int j = 0; j<width; j++)
				{
					Scalar1 = CV_RGB(0, 0, 0);
					cvSet2D(Image1, i, j, Scalar1);
					X = cvGet2D(warp_mat_N, 0, 0).val[0] * (j)+cvGet2D(warp_mat_N, 0, 1).val[0] * (i)+cvGet2D(warp_mat_N, 0, 2).val[0];
					Y = cvGet2D(warp_mat_N, 1, 0).val[0] * (j)+cvGet2D(warp_mat_N, 1, 1).val[0] * (i)+cvGet2D(warp_mat_N, 1, 2).val[0];
					T = cvGet2D(warp_mat_N, 2, 0).val[0] * (j)+cvGet2D(warp_mat_N, 2, 1).val[0] * (i)+cvGet2D(warp_mat_N, 2, 2).val[0];
					X = X / T;
					Y = Y / T;
					if (Y<height2 && Y >= 0 && X<width2 && X >= 0){
						Scalar2 = cvGet2D(src1, Y, X);
						Scalar1 = CV_RGB(Scalar2.val[2], Scalar2.val[1], Scalar2.val[0]);
						cvSet2D(Image1, i, j, Scalar1);
					}
				}
			}

			IplImage * src2 = cvLoadImage(img2_file);
			//cvWarpPerspective( src2, Image2, warp_mat );
			for (int i = 0; i<height; i++)
			{
				for (int j = 0; j<width; j++)
				{
					Scalar1 = CV_RGB(0, 0, 0);
					cvSet2D(Image2, i, j, Scalar1);
					X = cvGet2D(warp_mat_N, 0, 0).val[0] * (j)+cvGet2D(warp_mat_N, 0, 1).val[0] * (i)+cvGet2D(warp_mat_N, 0, 2).val[0];
					Y = cvGet2D(warp_mat_N, 1, 0).val[0] * (j)+cvGet2D(warp_mat_N, 1, 1).val[0] * (i)+cvGet2D(warp_mat_N, 1, 2).val[0];
					T = cvGet2D(warp_mat_N, 2, 0).val[0] * (j)+cvGet2D(warp_mat_N, 2, 1).val[0] * (i)+cvGet2D(warp_mat_N, 2, 2).val[0];
					X = X / T;
					Y = Y / T;
					if (Y<height1 && Y >= 0 && X<width1 && X >= 0){
						Scalar2 = cvGet2D(src2, Y, X);
						Scalar1 = CV_RGB(Scalar2.val[2], Scalar2.val[1], Scalar2.val[0]);
						cvSet2D(Image2, i, j, Scalar1);
					}
				}
			}
			/*cvShowImage("image2 MID", Image2);
			cvWaitKey(0);*/

			srcTri[0].x = keypoints2[min1].pt.x + tmp.cols*0.25;
			srcTri[0].y = keypoints2[min1].pt.y + tmp.rows*0.25;
			srcTri[1].x = keypoints2[min2].pt.x + tmp.cols*0.25;
			srcTri[1].y = keypoints2[min2].pt.y + tmp.rows*0.25;
			srcTri[2].x = keypoints2[min3].pt.x + tmp.cols*0.25;
			srcTri[2].y = keypoints2[min3].pt.y + tmp.rows*0.25;
			srcTri[3].x = keypoints2[min4].pt.x + tmp.cols*0.25;
			srcTri[3].y = keypoints2[min4].pt.y + tmp.rows*0.25;

			dstTri[0].x = keypoints1[p1].pt.x + tmp.cols*0.25;
			dstTri[0].y = keypoints1[p1].pt.y + tmp.rows*0.25;
			dstTri[1].x = keypoints1[p2].pt.x + tmp.cols*0.25;
			dstTri[1].y = keypoints1[p2].pt.y + tmp.rows*0.25;
			dstTri[2].x = keypoints1[p3].pt.x + tmp.cols*0.25;
			dstTri[2].y = keypoints1[p3].pt.y + tmp.rows*0.25;
			dstTri[3].x = keypoints1[p4].pt.x + tmp.cols*0.25;
			dstTri[3].y = keypoints1[p4].pt.y + tmp.rows*0.25;

			//cvGetPerspectiveTransform( srcTri, dstTri, warp_mat );
			float Array3[] = { srcTri[0].x, 0, srcTri[1].x, 0, srcTri[2].x, 0, srcTri[3].x, 0, srcTri[0].y, 0, srcTri[1].y, 0, srcTri[2].y, 0, srcTri[3].y, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, srcTri[0].x, 0, srcTri[1].x, 0, srcTri[2].x, 0, srcTri[3].x, 0, srcTri[0].y, 0, srcTri[1].y, 0, srcTri[2].y, 0, srcTri[3].y, 0, 1, 0, 1, 0, 1, 0, 1, -dstTri[0].x*srcTri[0].x, -dstTri[0].y*srcTri[0].x, -dstTri[1].x*srcTri[1].x, -dstTri[1].y*srcTri[1].x, -dstTri[2].x*srcTri[2].x, -dstTri[2].y*srcTri[2].x, -dstTri[3].x*srcTri[3].x, -dstTri[3].y*srcTri[3].x, -dstTri[0].x*srcTri[0].y, -dstTri[0].y*srcTri[0].y, -dstTri[1].x*srcTri[1].y, -dstTri[1].y*srcTri[1].y, -dstTri[2].x*srcTri[2].y, -dstTri[2].y*srcTri[2].y, -dstTri[3].x*srcTri[3].y, -dstTri[3].y*srcTri[3].y };
			cvSetData(TransposeMatrix, Array3, TransposeMatrix->step);
			float Array4[] = { dstTri[0].x, dstTri[0].y, dstTri[1].x, dstTri[1].y, dstTri[2].x, dstTri[2].y, dstTri[3].x, dstTri[3].y };
			cvSetData(Matrix2, Array4, Matrix2->step);
			cvTranspose(TransposeMatrix, Matrix1);
			/*cvMatMul(TransposeMatrix,Matrix1,ResultMatrix);
			cvInvert(ResultMatrix,InverseMatrix,CV_SVD);
			cvMatMul(InverseMatrix,TransposeMatrix,ResultMatrix);*/
			cvInvert(Matrix1, ResultMatrix, CV_SVD);//
			cvMatMul(ResultMatrix, Matrix2, FinalMatrix);
			cvmSet(warp_mat, 0, 0, cvmGet(FinalMatrix, 0, 0));
			cvmSet(warp_mat, 0, 1, cvmGet(FinalMatrix, 1, 0));
			cvmSet(warp_mat, 0, 2, cvmGet(FinalMatrix, 2, 0));
			cvmSet(warp_mat, 1, 0, cvmGet(FinalMatrix, 3, 0));
			cvmSet(warp_mat, 1, 1, cvmGet(FinalMatrix, 4, 0));
			cvmSet(warp_mat, 1, 2, cvmGet(FinalMatrix, 5, 0));
			cvmSet(warp_mat, 2, 0, cvmGet(FinalMatrix, 6, 0));
			cvmSet(warp_mat, 2, 1, cvmGet(FinalMatrix, 7, 0));
			cvmSet(warp_mat, 2, 2, 1);
			cvInvert(warp_mat, warp_mat_N, CV_SVD);
			//PrintMatrix(warp_mat_N,warp_mat_N->rows,warp_mat_N->cols);

			//Image3=Image2;
			for (int i = 0; i<height; i++)
			{
				for (int j = 0; j<width; j++)
				{
					Scalar1 = CV_RGB(0, 0, 0);
					cvSet2D(Image3, i, j, Scalar1);
					Scalar2 = cvGet2D(Image2, i, j);
					Scalar1 = CV_RGB(Scalar2.val[2], Scalar2.val[1], Scalar2.val[0]);
					//Scalar1=CV_RGB(255,0,0);
					cvSet2D(Image3, i, j, Scalar1);
				}
			}
			//cvWarpPerspective( Image3, Image2, warp_mat );
			for (int i = 0; i<height; i++)
			{
				for (int j = 0; j<width; j++)
				{
					Scalar1 = CV_RGB(0, 0, 0);
					cvSet2D(Image2, i, j, Scalar1);
					X = cvGet2D(warp_mat_N, 0, 0).val[0] * (j)+cvGet2D(warp_mat_N, 0, 1).val[0] * (i)+cvGet2D(warp_mat_N, 0, 2).val[0];
					Y = cvGet2D(warp_mat_N, 1, 0).val[0] * (j)+cvGet2D(warp_mat_N, 1, 1).val[0] * (i)+cvGet2D(warp_mat_N, 1, 2).val[0];
					T = cvGet2D(warp_mat_N, 2, 0).val[0] * (j)+cvGet2D(warp_mat_N, 2, 1).val[0] * (i)+cvGet2D(warp_mat_N, 2, 2).val[0];
					X = X / T;
					Y = Y / T;
					if (Y<height && Y >= 0 && X<width && X >= 0){
						//cout<<endl<<"X:"<<X<<" Y:"<<Y<<endl;
						Scalar2 = cvGet2D(Image3, Y, X);
						Scalar1 = CV_RGB(Scalar2.val[2], Scalar2.val[1], Scalar2.val[0]);
						//Scalar1=CV_RGB(255,0,0);
						cvSet2D(Image2, i, j, Scalar1);
					}
				}
			}
			cvInvert(warp_mat, warp_mat_N, CV_SVD);
			in_num = 0;
			for (int i = 0; i<key1; ++i){
				if (valid[i] == 1){
					X = cvGet2D(warp_mat_N, 0, 0).val[0] * (keypoints1[i].pt.x + tmp.cols*0.25) + cvGet2D(warp_mat_N, 0, 1).val[0] * (keypoints1[i].pt.y + tmp.rows*0.25) + cvGet2D(warp_mat_N, 0, 2).val[0];
					Y = cvGet2D(warp_mat_N, 1, 0).val[0] * (keypoints1[i].pt.x + tmp.cols*0.25) + cvGet2D(warp_mat_N, 1, 1).val[0] * (keypoints1[i].pt.y + tmp.rows*0.25) + cvGet2D(warp_mat_N, 1, 2).val[0];
					T = cvGet2D(warp_mat_N, 2, 0).val[0] * (keypoints1[i].pt.x + tmp.cols*0.25) + cvGet2D(warp_mat_N, 2, 1).val[0] * (keypoints1[i].pt.y + tmp.rows*0.25) + cvGet2D(warp_mat_N, 2, 2).val[0];
					X = X / T;
					Y = Y / T;

					dis = (
						(keypoints2[ivector[i]].pt.x + tmp.cols*0.25) - X)*((keypoints2[ivector[i]].pt.x + tmp.cols*0.25) - X) + ((keypoints2[ivector[i]].pt.y + tmp.rows*0.25) - Y)*((keypoints2[ivector[i]].pt.y + tmp.rows*0.25) - Y
					);

					if (dis<RANSAC_limit_pixel)
						in_num++;
				}
			}
			//cout << endl << in_num << endl;
			warp = warp_mat_N;
			RANSAC_counter++;
		}
		std::cout << " RANSAC iteration : " << RANSAC_counter << std::endl;
		// end RANSAC
		//cout << "RANSAC time : " << (double)(clock() - tStart) / CLOCKS_PER_SEC << endl;
		// Blend images
		if (RANSAC_counter < RANSAC_limit){
			for (int i = 0; i<height; i++){
				for (int j = 0; j<width; j++){
					X = cvGet2D(warp, 0, 0).val[0] * (j)+cvGet2D(warp, 0, 1).val[0] * (i)+cvGet2D(warp, 0, 2).val[0];
					Y = cvGet2D(warp, 1, 0).val[0] * (j)+cvGet2D(warp, 1, 1).val[0] * (i)+cvGet2D(warp, 1, 2).val[0];
					T = cvGet2D(warp, 2, 0).val[0] * (j)+cvGet2D(warp, 2, 1).val[0] * (i)+cvGet2D(warp, 2, 2).val[0];
					X = X / T;
					Y = Y / T;

					Scalar1 = cvGet2D(Image1, i, j);
					Scalar2 = cvGet2D(Image2, i, j);

					if (X>tmp.cols*0.25 + 1 && X<width1 + tmp.cols*0.25 - 2 && Y>tmp.rows*0.25 + 1 && Y<height1 + tmp.rows*0.25 - 2){
						Scalar1 = CV_RGB(Scalar2.val[2], Scalar2.val[1], Scalar2.val[0]);
						cvSet2D(Image1, i, j, Scalar1);
					}
				}
			}
			//cvShowImage("Result Image", Image1);
			//cvWaitKey(0);
			
			strcpy(result, str1);
			strcat(result, "_");
			strcat(result, str2);
			char * filename = (char *)malloc(1 + strlen(result) + 4);
			strcpy(filename, result);
			strcat(filename, ".jpg");
			cvSaveImage(filename, Image1);
		}else{
			//cout << "No Matching Keypoints" << endl;
			std::cout << " RANSAC Failed " << std::endl;
			strcpy(result, "Failed!");
			return result;
		}
	}
	else{
		std::cout << " NO Matching Keypoints " << std::endl;
		//cout << "No Matching Keypoints" << endl;
		strcpy(result, "Failed!");
		return result;
	}

	return result;
}

void PrintMatrix(CvMat *Matrix, int Rows, int Cols){
	for (int i = 0; i<Rows; i++){
		for (int j = 0; j<Cols; j++){
			printf("%.1f ", cvGet2D(Matrix, i, j).val[0]);
		}
		printf("\n");
	}
}
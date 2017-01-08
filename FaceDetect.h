
#ifndef FACEDETECT_H
#define	FACEDETECT_H

#include "defs.h"
#include <string>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2\face.hpp>
#include <opencv2\flann\flann.hpp>
#include "preprocessFace.h"
#include "detectObject.h"

using namespace std;
using namespace cv;
using namespace face;
using namespace flann;

class FaceDetect {
public:
	FaceDetect(
		double scaleFactor,
		int    minNeighbors,
		double minSizeRatio,
		double maxSizeRatio,
		int width, int height);
	virtual ~FaceDetect();
	void findFacesInImage(const Mat &img, vector<Rect> &res);
	Mat normalize(Mat img);
	Mat normalize(Mat img, bool isobject);
	Mat normalizeImage(const Mat &img, bool isobject);
	void initDetectors();
	void detectBothEyes(const Mat &face, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, Point &leftEye, Point &rightEye, Rect *searchedLeftEye, Rect *searchedRightEye);
	void equalizeLeftAndRightHalves(Mat &faceImg);
	void findEyes(const Mat &img, vector<Rect> &res);
	Mat getPreprocessedFace(Mat &srcImg, int desiredFaceWidth, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, bool doLeftAndRightSeparately, Rect *storeFaceRect, Point *storeLeftEye, Point *storeRightEye, Rect *searchedLeftEye, Rect *searchedRightEye);
	Mat norm_0_255(const Mat& src);
	Mat preProcessImage(InputArray src,
		float alpha, float tau, float gamma, int sigma0, int sigma1);
private:
	const double DESIRED_LEFT_EYE_X = 0.16;     // Controls how much of the face is visible after preprocessing.
	const double DESIRED_LEFT_EYE_Y = 0.14;
	const double FACE_ELLIPSE_CY = 0.40;
	const double FACE_ELLIPSE_W = 0.50;         // Should be atleast 0.5
	const double FACE_ELLIPSE_H = 0.80;         // Controls how tall the face mask is.

	CascadeClassifier _faceCascade;
	CascadeClassifier _eyeCascade;
	
	double _scaleFactor;
	int    _minNeighbors;
	int	   _width;
	int	   _height;
	double _minSizeRatio;
	double _maxSizeRatio;
};

#endif	/* FACEDETECT_H */


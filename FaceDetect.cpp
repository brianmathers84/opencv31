#include <vector>
#include "stdafx.h"
#include "FaceDetect.h"
FaceDetect::FaceDetect(
	double scaleFactor,
	int    minNeighbors,
	double minSizeRatio,
	double maxSizeRatio, int width, int height) :
	_scaleFactor(scaleFactor), _minNeighbors(minNeighbors), _minSizeRatio(minSizeRatio), _maxSizeRatio(maxSizeRatio) {
	_width = width;
	_height = height;
	initDetectors();
}

FaceDetect::~FaceDetect() {}
Mat FaceDetect::normalize(Mat img) {
	resize(img, img, Size(_width, _height));
	cvtColor(img, img, CV_BGR2GRAY);
	equalizeHist(img, img);
	return img;
}
Mat FaceDetect::normalize(Mat input_img, bool obj) {
	Mat src_gray;
	Mat detected_edges;
	src_gray.create(Size(input_img.cols, input_img.rows), CV_8UC1);
	//cvtColor(input_img, src_gray, CV_BGR2GRAY);
	return input_img;
//	blur(src_gray, detected_edges, Size(3, 3));
//	Canny(detected_edges, detected_edges, 1, 100, 3);
//	return detected_edges;
}
Mat FaceDetect::normalizeImage(const Mat &img, bool isobject) {
	Mat tmp;
	//tmp = norm_0_255(img);
	//tmp = preProcessImage(img, 0.1, 10.0, 0.2, 1, 2);
	if (isobject)
		tmp = normalize(img, true);
	else tmp = normalize(img);
	
	return tmp;
}
void FaceDetect::initDetectors()
{
	_faceCascade.load(FACE_DETECT);
	_eyeCascade.load(EYE_DETECT);
}

void FaceDetect::findFacesInImage(const Mat &img, vector<Rect> &res) {		
	//_width = img.size().width,
	//_height = img.size().height;
	Size minScaleSize = Size(_minSizeRatio  * _width, _minSizeRatio  * _height),
		maxScaleSize = Size(_maxSizeRatio  * _width, _maxSizeRatio  * _height);	
	res.clear();

	//detect faces:
	_faceCascade.detectMultiScale(img, res, _scaleFactor, _minNeighbors, 0, minScaleSize, maxScaleSize);
}
void FaceDetect::findEyes(const Mat &img, vector<Rect> &res) {
	_eyeCascade.detectMultiScale(img, res, _scaleFactor, _minNeighbors, 0);
}
/*
//Step 1: detect landmarks over the detected face
vector<cv::Point2d> landmarks = landmark_detector->detectLandmarks(img_gray, Rect(r.x, r.y, r.width, r.height));
//Step 2: align face
Mat aligned_image;
vector<cv::Point2d> aligned_landmarks;
aligner->align(img_gray, aligned_image, landmarks, aligned_landmarks);
//Step 3: normalize region
Mat normalized_region = normalizer->process(aligned_image, Rect(r.x, r.y, r.width, r.height), aligned_landmarks);
//Step 4: tan&&triggs
normalized_region = ((FlandMarkFaceAlign *)normalizer)->tan_triggs_preprocessing(normalized_region, gamma_correct, dog, contrast_eq);
vector<cv::Point2d> FlandmarkLandmarkDetection::detectLandmarks(const Mat & image, const Rect & face) {

	vector<Point2d> landmarks;

	int bbox[4] = { face.x, face.y, face.x + face.width, face.y + face.height };
	double *points = new double[2 * this->model->data.options.M];

	//http://cmp.felk.cvut.cz/~uricamic/flandmark/
	if (flandmark_detect(new IplImage(image), bbox, this->model, points) > 0) {
		return landmarks;
	}

	for (int i = 0; i < this->model->data.options.M; i++) {
		landmarks.push_back(Point2f(points[2 * i], points[2 * i + 1]));
	}

	LinearRegression lr;
	lr.addPoint(Point2D(landmarks[LEFT_EYE_OUTER].x, landmarks[LEFT_EYE_OUTER].y));
	lr.addPoint(Point2D(landmarks[LEFT_EYE_INNER].x, landmarks[LEFT_EYE_INNER].y));
	lr.addPoint(Point2D(landmarks[RIGHT_EYE_INNER].x, landmarks[RIGHT_EYE_INNER].y));
	lr.addPoint(Point2D(landmarks[RIGHT_EYE_OUTER].x, landmarks[RIGHT_EYE_OUTER].y));

	double coef_determination = lr.getCoefDeterm();
	double coef_correlation = lr.getCoefCorrel();
	double standar_error_estimate = lr.getStdErrorEst();

	double a = lr.getA();
	double b = lr.getB();

	cv::Point pp1(landmarks[LEFT_EYE_OUTER].x, landmarks[LEFT_EYE_OUTER].x*b + a);
	cv::Point pp2(landmarks[RIGHT_EYE_OUTER].x, landmarks[RIGHT_EYE_OUTER].x*b + a);

	landmarks.push_back(pp1); //landmarks[LEFT_EYE_ALIGN]
	landmarks.push_back(pp2); //landmarks[RIGHT_EYE_ALIGN]

	delete[] points;
	points = 0;
	return landmarks;
}
*/
											/*
											// Remove the outer border of the face, so it doesn't include the background & hair.
											// Keeps the center of the rectangle at the same place, rather than just dividing all values by 'scale'.
											Rect scaleRectFromCenter(const Rect wholeFaceRect, float scale)
											{
											float faceCenterX = wholeFaceRect.x + wholeFaceRect.width * 0.5f;
											float faceCenterY = wholeFaceRect.y + wholeFaceRect.height * 0.5f;
											float newWidth = wholeFaceRect.width * scale;
											float newHeight = wholeFaceRect.height * scale;
											Rect faceRect;
											faceRect.width = cvRound(newWidth);                        // Shrink the region
											faceRect.height = cvRound(newHeight);
											faceRect.x = cvRound(faceCenterX - newWidth * 0.5f);    // Move the region so that the center is still the same spot.
											faceRect.y = cvRound(faceCenterY - newHeight * 0.5f);

											return faceRect;
											}
											*/

											// Search for both eyes within the given face image. Returns the eye centers in 'leftEye' and 'rightEye',
											// or sets them to (-1,-1) if each eye was not found. Note that you can pass a 2nd eyeCascade if you
											// want to search eyes using 2 different cascades. For example, you could use a regular eye detector
											// as well as an eyeglasses detector, or a left eye detector as well as a right eye detector.
											// Or if you don't want a 2nd eye detection, just pass an uninitialized CascadeClassifier.
											// Can also store the searched left & right eye regions if desired.
void FaceDetect::detectBothEyes(const Mat &face, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, Point &leftEye, Point &rightEye, Rect *searchedLeftEye, Rect *searchedRightEye)
{
	// Skip the borders of the face, since it is usually just hair and ears, that we don't care about.
	/*
	// For "2splits.xml": Finds both eyes in roughly 60% of detected faces, also detects closed eyes.
	const float EYE_SX = 0.12f;
	const float EYE_SY = 0.17f;
	const float EYE_SW = 0.37f;
	const float EYE_SH = 0.36f;
	*/
	/*
	// For mcs.xml: Finds both eyes in roughly 80% of detected faces, also detects closed eyes.
	const float EYE_SX = 0.10f;
	const float EYE_SY = 0.19f;
	const float EYE_SW = 0.40f;
	const float EYE_SH = 0.36f;
	*/

	// For default eye.xml or eyeglasses.xml: Finds both eyes in roughly 40% of detected faces, but does not detect closed eyes.
	const float EYE_SX = 0.16f;
	const float EYE_SY = 0.26f;
	const float EYE_SW = 0.30f;
	const float EYE_SH = 0.28f;

	int leftX = cvRound(face.cols * EYE_SX);
	int topY = cvRound(face.rows * EYE_SY);
	int widthX = cvRound(face.cols * EYE_SW);
	int heightY = cvRound(face.rows * EYE_SH);
	int rightX = cvRound(face.cols * (1.0 - EYE_SX - EYE_SW));  // Start of right-eye corner

	Mat topLeftOfFace = face(Rect(leftX, topY, widthX, heightY));
	Mat topRightOfFace = face(Rect(rightX, topY, widthX, heightY));
	Rect leftEyeRect, rightEyeRect;

	// Return the search windows to the caller, if desired.
	if (searchedLeftEye)
		*searchedLeftEye = Rect(leftX, topY, widthX, heightY);
	if (searchedRightEye)
		*searchedRightEye = Rect(rightX, topY, widthX, heightY);

	// Search the left region, then the right region using the 1st eye detector.
	detectLargestObject(topLeftOfFace, eyeCascade1, leftEyeRect, topLeftOfFace.cols);
	detectLargestObject(topRightOfFace, eyeCascade1, rightEyeRect, topRightOfFace.cols);

	// If the eye was not detected, try a different cascade classifier.
	if (leftEyeRect.width <= 0 && !eyeCascade2.empty()) {
		detectLargestObject(topLeftOfFace, eyeCascade2, leftEyeRect, topLeftOfFace.cols);
		//if (leftEyeRect.width > 0)
		//    cout << "2nd eye detector LEFT SUCCESS" << endl;
		//else
		//    cout << "2nd eye detector LEFT failed" << endl;
	}
	//else
	//    cout << "1st eye detector LEFT SUCCESS" << endl;

	// If the eye was not detected, try a different cascade classifier.
	if (rightEyeRect.width <= 0 && !eyeCascade2.empty()) {
		detectLargestObject(topRightOfFace, eyeCascade2, rightEyeRect, topRightOfFace.cols);
		//if (rightEyeRect.width > 0)
		//    cout << "2nd eye detector RIGHT SUCCESS" << endl;
		//else
		//    cout << "2nd eye detector RIGHT failed" << endl;
	}
	//else
	//    cout << "1st eye detector RIGHT SUCCESS" << endl;

	if (leftEyeRect.width > 0) {   // Check if the eye was detected.
		leftEyeRect.x += leftX;    // Adjust the left-eye rectangle because the face border was removed.
		leftEyeRect.y += topY;
		leftEye = Point(leftEyeRect.x + leftEyeRect.width / 2, leftEyeRect.y + leftEyeRect.height / 2);
	}
	else {
		leftEye = Point(-1, -1);    // Return an invalid point
	}

	if (rightEyeRect.width > 0) { // Check if the eye was detected.
		rightEyeRect.x += rightX; // Adjust the right-eye rectangle, since it starts on the right side of the image.
		rightEyeRect.y += topY;  // Adjust the right-eye rectangle because the face border was removed.
		rightEye = Point(rightEyeRect.x + rightEyeRect.width / 2, rightEyeRect.y + rightEyeRect.height / 2);
	}
	else {
		rightEye = Point(-1, -1);    // Return an invalid point
	}
}

// Histogram Equalize seperately for the left and right sides of the face.
void FaceDetect::equalizeLeftAndRightHalves(Mat &faceImg)
{
	// It is common that there is stronger light from one half of the face than the other. In that case,
	// if you simply did histogram equalization on the whole face then it would make one half dark and
	// one half bright. So we will do histogram equalization separately on each face half, so they will
	// both look similar on average. But this would cause a sharp edge in the middle of the face, because
	// the left half and right half would be suddenly different. So we also histogram equalize the whole
	// image, and in the middle part we blend the 3 images together for a smooth brightness transition.

	int w = faceImg.cols;
	int h = faceImg.rows;

	// 1) First, equalize the whole face.
	Mat wholeFace;
	equalizeHist(faceImg, wholeFace);

	// 2) Equalize the left half and the right half of the face separately.
	int midX = w / 2;
	Mat leftSide = faceImg(Rect(0, 0, midX, h));
	Mat rightSide = faceImg(Rect(midX, 0, w - midX, h));
	equalizeHist(leftSide, leftSide);
	equalizeHist(rightSide, rightSide);

	// 3) Combine the left half and right half and whole face together, so that it has a smooth transition.
	for (int y = 0; y<h; y++) {
		for (int x = 0; x<w; x++) {
			int v;
			if (x < w / 4) {          // Left 25%: just use the left face.
				v = leftSide.at<uchar>(y, x);
			}
			else if (x < w * 2 / 4) {   // Mid-left 25%: blend the left face & whole face.
				int lv = leftSide.at<uchar>(y, x);
				int wv = wholeFace.at<uchar>(y, x);
				// Blend more of the whole face as it moves further right along the face.
				float f = (x - w * 1 / 4) / (float)(w*0.25f);
				v = cvRound((1.0f - f) * lv + (f)* wv);
			}
			else if (x < w * 3 / 4) {   // Mid-right 25%: blend the right face & whole face.
				int rv = rightSide.at<uchar>(y, x - midX);
				int wv = wholeFace.at<uchar>(y, x);
				// Blend more of the right-side face as it moves further right along the face.
				float f = (x - w * 2 / 4) / (float)(w*0.25f);
				v = cvRound((1.0f - f) * wv + (f)* rv);
			}
			else {                  // Right 25%: just use the right face.
				v = rightSide.at<uchar>(y, x - midX);
			}
			faceImg.at<uchar>(y, x) = v;
		}// end x loop
	}//end y loop
}


// Create a grayscale face image that has a standard size and contrast & brightness.
// "srcImg" should be a copy of the whole color camera frame, so that it can draw the eye positions onto.
// If 'doLeftAndRightSeparately' is true, it will process left & right sides seperately,
// so that if there is a strong light on one side but not the other, it will still look OK.
// Performs Face Preprocessing as a combination of:
//  - geometrical scaling, rotation and translation using Eye Detection,
//  - smoothing away image noise using a Bilateral Filter,
//  - standardize the brightness on both left and right sides of the face independently using separated Histogram Equalization,
//  - removal of background and hair using an Elliptical Mask.
// Returns either a preprocessed face square image or NULL (ie: couldn't detect the face and 2 eyes).
// If a face is found, it can store the rect coordinates into 'storeFaceRect' and 'storeLeftEye' & 'storeRightEye' if given,
// and eye search regions into 'searchedLeftEye' & 'searchedRightEye' if given.
Mat FaceDetect::getPreprocessedFace(Mat &srcImg, int desiredFaceWidth, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, bool doLeftAndRightSeparately, Rect *storeFaceRect, Point *storeLeftEye, Point *storeRightEye, Rect *searchedLeftEye, Rect *searchedRightEye)
{
	// Use square faces.
	int desiredFaceHeight = desiredFaceWidth;

	// Mark the detected face region and eye search regions as invalid, in case they aren't detected.
	if (storeFaceRect)
		storeFaceRect->width = -1;
	if (storeLeftEye)
		storeLeftEye->x = -1;
	if (storeRightEye)
		storeRightEye->x = -1;
	if (searchedLeftEye)
		searchedLeftEye->width = -1;
	if (searchedRightEye)
		searchedRightEye->width = -1;

	// Find the largest face.
	Rect faceRect;
	detectLargestObject(srcImg, faceCascade, faceRect);

	// Check if a face was detected.
	if (faceRect.width > 0) {

		// Give the face rect to the caller if desired.
		if (storeFaceRect)
			*storeFaceRect = faceRect;

		Mat faceImg = srcImg(faceRect);    // Get the detected face image.

										   // If the input image is not grayscale, then convert the BGR or BGRA color image to grayscale.
		Mat gray;
		if (faceImg.channels() == 3) {
			cvtColor(faceImg, gray, CV_BGR2GRAY);
		}
		else if (faceImg.channels() == 4) {
			cvtColor(faceImg, gray, CV_BGRA2GRAY);
		}
		else {
			// Access the input image directly, since it is already grayscale.
			gray = faceImg;
		}

		// Search for the 2 eyes at the full resolution, since eye detection needs max resolution possible!
		Point leftEye, rightEye;
		detectBothEyes(gray, eyeCascade1, eyeCascade2, leftEye, rightEye, searchedLeftEye, searchedRightEye);

		// Give the eye results to the caller if desired.
		if (storeLeftEye)
			*storeLeftEye = leftEye;
		if (storeRightEye)
			*storeRightEye = rightEye;

		// Check if both eyes were detected.
		if (leftEye.x >= 0 && rightEye.x >= 0) {

			// Make the face image the same size as the training images.

			// Since we found both eyes, lets rotate & scale & translate the face so that the 2 eyes
			// line up perfectly with ideal eye positions. This makes sure that eyes will be horizontal,
			// and not too far left or right of the face, etc.

			// Get the center between the 2 eyes.
			Point2f eyesCenter = Point2f((leftEye.x + rightEye.x) * 0.5f, (leftEye.y + rightEye.y) * 0.5f);
			// Get the angle between the 2 eyes.
			double dy = (rightEye.y - leftEye.y);
			double dx = (rightEye.x - leftEye.x);
			double len = sqrt(dx*dx + dy*dy);
			double angle = atan2(dy, dx) * 180.0 / CV_PI; // Convert from radians to degrees.

														  // Hand measurements shown that the left eye center should ideally be at roughly (0.19, 0.14) of a scaled face image.
			double DESIRED_RIGHT_EYE_X = (1.0f - DESIRED_LEFT_EYE_X);
			// Get the amount we need to scale the image to be the desired fixed size we want.
			double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * desiredFaceWidth;
			double scale = desiredLen / len;
			// Get the transformation matrix for rotating and scaling the face to the desired angle & size.
			Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
			// Shift the center of the eyes to be the desired center between the eyes.
			rot_mat.at<double>(0, 2) += desiredFaceWidth * 0.5f - eyesCenter.x;
			rot_mat.at<double>(1, 2) += desiredFaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter.y;

			// Rotate and scale and translate the image to the desired angle & size & position!
			// Note that we use 'w' for the height instead of 'h', because the input face has 1:1 aspect ratio.
			Mat warped = Mat(desiredFaceHeight, desiredFaceWidth, CV_8U, Scalar(128)); // Clear the output image to a default grey.
			warpAffine(gray, warped, rot_mat, warped.size());
			//imshow("warped", warped);

			// Give the image a standard brightness and contrast, in case it was too dark or had low contrast.
			if (!doLeftAndRightSeparately) {
				// Do it on the whole face.
				equalizeHist(warped, warped);
			}
			else {
				// Do it seperately for the left and right sides of the face.
				equalizeLeftAndRightHalves(warped);
			}
			//imshow("equalized", warped);

			// Use the "Bilateral Filter" to reduce pixel noise by smoothing the image, but keeping the sharp edges in the face.
			Mat filtered = Mat(warped.size(), CV_8U);
			bilateralFilter(warped, filtered, 0, 20.0, 2.0);
			//imshow("filtered", filtered);

			// Filter out the corners of the face, since we mainly just care about the middle parts.
			// Draw a filled ellipse in the middle of the face-sized image.
			Mat mask = Mat(warped.size(), CV_8U, Scalar(0)); // Start with an empty mask.
			Point faceCenter = Point(desiredFaceWidth / 2, 
				cvRound(desiredFaceHeight * FACE_ELLIPSE_CY));
			Size size = Size(cvRound(desiredFaceWidth * FACE_ELLIPSE_W), cvRound(desiredFaceHeight * FACE_ELLIPSE_H));
			ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);
			//imshow("mask", mask);

			// Use the mask, to remove outside pixels.
			Mat dstImg = Mat(warped.size(), CV_8U, Scalar(128)); // Clear the output image to a default gray.
																 /*
																 namedWindow("filtered");
																 imshow("filtered", filtered);
																 namedWindow("dstImg");
																 imshow("dstImg", dstImg);
																 namedWindow("mask");
																 imshow("mask", mask);
																 */
																 // Apply the elliptical mask on the face.
			filtered.copyTo(dstImg, mask);  // Copies non-masked pixels from filtered to dstImg.
											//imshow("dstImg", dstImg);

			return dstImg;
		}
		/*
		else {
		// Since no eyes were found, just do a generic image resize.
		resize(gray, tmpImg, Size(w,h));
		}
		*/
	}
	return Mat();
}
// Normalizes a given image into a value range between 0 and 255.
Mat FaceDetect::norm_0_255(const Mat& src) {
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}
//
// Calculates the TanTriggs Preprocessing as described in:
//
//      Tan, X., and Triggs, B. "Enhanced local texture feature sets for face
//      recognition under difficult lighting conditions.". IEEE Transactions
//      on Image Processing 19 (2010), 1635–650.
//
// Default parameters are taken from the paper.
//
Mat FaceDetect::preProcessImage(InputArray src,
	float alpha = 0.1, float tau = 10.0, float gamma = 0.2, int sigma0 = 1,
	int sigma1 = 2) {

	// Convert to floating point:
	Mat X = src.getMat();
	X.convertTo(X, CV_32FC1);
	// Start preprocessing:
	Mat I;
	pow(X, gamma, I);
	// Calculate the DOG Image:
	{
		Mat gaussian0, gaussian1;
		// Kernel Size:
		int kernel_sz0 = (3 * sigma0);
		int kernel_sz1 = (3 * sigma1);
		// Make them odd for OpenCV:
		kernel_sz0 += ((kernel_sz0 % 2) == 0) ? 1 : 0;
		kernel_sz1 += ((kernel_sz1 % 2) == 0) ? 1 : 0;
		GaussianBlur(I, gaussian0, Size(kernel_sz0, kernel_sz0), sigma0, sigma0, BORDER_REPLICATE);
		GaussianBlur(I, gaussian1, Size(kernel_sz1, kernel_sz1), sigma1, sigma1, BORDER_REPLICATE);
		subtract(gaussian0, gaussian1, I);
	}

	{
		double meanI = 0.0;
		{
			Mat tmp;
			pow(abs(I), alpha, tmp);
			meanI = mean(tmp).val[0];

		}
		I = I / pow(meanI, 1.0 / alpha);
	}

	{
		double meanI = 0.0;
		{
			Mat tmp;
			pow(min(abs(I), tau), alpha, tmp);
			meanI = mean(tmp).val[0];
		}
		I = I / pow(meanI, 1.0 / alpha);
	}

	// Squash into the tanh:
	{
		Mat exp_x, exp_negx;
		exp(I / tau, exp_x);
		exp(-I / tau, exp_negx);
		divide(exp_x - exp_negx, exp_x + exp_negx, I);
		I = tau * I;
	}
	I.convertTo(I, CV_8UC1);
	return I;
}
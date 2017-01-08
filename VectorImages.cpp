#include "stdafx.h"
#include "VectorImages.h"
#include "defs.h"

VectorImages::VectorImages()
{
}

Mat VectorImages::normalize(Mat img) {
	resize(img, img, Size(DET_WIDTH, DET_HEIGHT));
	cvtColor(img, img, CV_BGR2GRAY);
	equalizeHist(img, img);
	return img;
}

VectorImages::VectorImages(vector<Mat> faces, vector<PeopleImages> peoples, vector<int> imageIDs) {
	_faces = faces;
	_peoples = peoples;
	_imageIDs = imageIDs;
}

VectorImages::~VectorImages()
{
//	delete _faces;
}

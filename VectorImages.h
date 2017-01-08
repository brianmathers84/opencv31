#ifndef VECTORIMAGES_H
#define	VECTORIMAGES_H
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include "Person.h"
#include "PeopleImages.h"
#include "FaceDetect.h"

using namespace cv;
using namespace std;
using namespace face;
class VectorImages
{
public:
	vector<PeopleImages> _peoples;
	vector<Mat, PeopleImages> _allPeoples;
	vector<Mat> _faces;
	vector<int> _imageIDs;
	VectorImages();
	VectorImages(vector<Mat> faces, vector<PeopleImages> peoples, vector<int> imageIDs);
	Mat normalize(Mat img);
	virtual ~VectorImages();
};
#endif	/* VECTORIMAGES_H */


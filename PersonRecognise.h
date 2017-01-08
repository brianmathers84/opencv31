#ifndef PersonRecognise_H
#define	PersonRecognise_H

#define PERSON_LABEL 10 //some arbitrary label

#include <string>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2\face.hpp>

using namespace std;
using namespace cv;
using namespace face;
enum PersonType {FAMILY,RAFF,OTHER};
class PersonRecognise {
public:
	PersonRecognise(const vector<Mat> &imgs, vector<int> ids, int radius, int neighbors,
		int size_w, int size_h, int grid_x, int grid_y, double threshold);
	PersonRecognise(const vector<Mat> &imgs, vector<int> ids, int radius, int neighbors,
		int size_w, int size_h, int grid_x, int grid_y, double threshold, PersonType s_type);
	bool detectObject(const Mat &face, Mat_<Vec4f> &pos);
	~PersonRecognise();
	bool recognize(const Mat &face, int &userid, double &confidence) const;
private:
	vector<int> _ids;
	Ptr<FaceRecognizer> _model;
	Ptr<GeneralizedHoughBallard> _imodel;
	Size _faceSize;
	double _threshold;
};

#endif	/* PersonRecognise_H */


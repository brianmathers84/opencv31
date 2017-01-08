#include "stdafx.h"
#include "PersonRecognise.h"
PersonRecognise::PersonRecognise(const vector<Mat> &imgs, vector<int> ids, int radius, int neighbors,
	int size_w, int size_h, int grid_x, int grid_y, double threshold) {
	_ids = ids;
	_threshold = threshold;
	_faceSize = Size(size_w, size_h);
	//build recognizer model:
//	_model = createLBPHFaceRecognizer();// radius, neighbors, grid_x, grid_y, 0);
	_model = createFisherFaceRecognizer();// 0, threshold);
//	_model = createEigenFaceRecognizer(0, threshold);
	_model->train(imgs, ids);
}
PersonRecognise::PersonRecognise(const vector<Mat> &imgs, vector<int> ids, int radius, int neighbors,
	int size_w, int size_h, int grid_x, int grid_y, double threshold, PersonType s_type) {
	_ids = ids;
	_threshold = threshold;
	_faceSize = Size(size_w, size_h);
	//build recognizer model:
	switch (s_type) {
		case FAMILY: _model = createEigenFaceRecognizer(); 	
			_model->train(imgs, ids); break;
		case RAFF: _imodel = createGeneralizedHoughBallard(); 
//			_imodel->setTemplate(imgs[0]);
			break;
		default:_model = createEigenFaceRecognizer(); 
			_model->train(imgs, ids); break;
	}
}
bool PersonRecognise::detectObject(const Mat &face, Mat_<Vec4f> &pos) {
	_imodel->detect(face, pos);
	return (pos.empty() == true);
}
PersonRecognise::~PersonRecognise() {}

bool PersonRecognise::recognize(const Mat &face, int &userid, double &confidence) const {
	Mat gray;
	int nStat=0;
	cvtColor(face, gray, CV_BGR2GRAY);
	resize(gray, gray, _faceSize);
	equalizeHist(gray, gray);
	_model->predict(gray, userid, confidence);
	_model->getLabelInfo(userid);
	return -1 != userid ? true : false;
}
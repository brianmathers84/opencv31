//#include <opencv\cv.h>   // This is the original code, but I couldn't get VideoCapture work correctly.
#include "stdafx.h"
#include <iostream>
#include <dirent.h>
#include <string.h>

#include "defs.h"
#include <fstream>
#include <locale>
#include <codecvt>
// sound
#include <windows.h>
#include <Mmsystem.h>
#include <mciapi.h>
#pragma comment(lib, "Winmm.lib")
// people / faces
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2\face.hpp>
#include "Person.h"
#include "PeopleImages.h"
#include "sqlCommands.h"
#include "FaceDetect.h"
#include "PersonRecognise.h"
#include <ctime>

using namespace cv;
using namespace std;
using namespace face;
vector<PeopleImages> peopleImages;
vector<Mat> images;
vector<int> filteredIds;
vector<Mat> objimages;
vector<int> objfilteredIds;
vector<pair<int,string>> filteredNames;
vector<pair<int, int>> personLookup;

sqlCommands sqlclass;

void getDBUsers() {
	sqlclass.readImages();	// read users and image filenames from DB
	peopleImages = sqlclass.getPeople();
}
void getSqlImageDB() {
	string imgName;
	FaceDetect fd( DET_SCALE_FACTOR, DET_MIN_NEIGHBORS,
		DET_MIN_SIZE_RATIO, DET_MAX_SIZE_RATIO, DET_WIDTH, DET_HEIGHT);
	vector<Rect> faces;
	int found = 0;
	for (int i = 0; i < peopleImages.size(); i++) {
		PeopleImages ppi = peopleImages.at(i);
		Person pp = ppi.getPerson();
		imgName = ppi._image_filename;
		try {
			Mat imgDb = imread(imgName);
			if (pp._isperson!=1) {
				Mat objimg = fd.normalizeImage(imgDb, true);
				objimages.push_back(objimg);
				objfilteredIds.push_back(found);
				pair<int, int> personPair = make_pair(found, pp._personid);
				personLookup.push_back(personPair);
				pair<int, string> imgPair = make_pair(found, pp._firstname + "+" + pp._surname);
				filteredNames.push_back(imgPair);
				found++;
				imshow("window", imgDb);	//brady bunch!
				waitKey(33);
			}
			else {
//				imgDb = fd.normalizeImage(imgDb, false);
				fd.findFacesInImage(imgDb, faces);
				//analyze each detected face:
				if (faces.empty()) {
					string imgs = imgName.substr(7, imgName.size() - 3);
					printf("\nrejected image: %s", imgs);
				}
				for (vector<Rect>::const_iterator face = faces.begin(); face != faces.end(); face++) {
					Scalar color = NO_MATCH_COLOR;
					Mat face_img = imgDb(*face);
					face_img = fd.normalizeImage(face_img, false);
					
					pair<int, string> imgPair = make_pair(found, pp._firstname+" "+pp._surname);
					filteredNames.push_back(imgPair);
					pair<int, int> personPair = make_pair(found, pp._personid);
					personLookup.push_back(personPair);
					images.push_back(face_img);
					filteredIds.push_back(found);
					found++;
					imshow("window", face_img);	//print image to screen				
					waitKey(33);
				}
			}
		}
		catch (Exception ex) {} // ignore file read errors
	}
}

void playsound(string filename) {
	string mcistring = "open \"" + filename + "\" type waveaudio alias audiofile";
	wstring wstr(mcistring.begin(), mcistring.end());
	mciSendString(wstr.c_str(), NULL, 0, 0);
	mcistring = "play audiofile from 0";
	wstring wstrplay(mcistring.begin(), mcistring.end());
	mciSendString(wstrplay.c_str(), NULL, 0, 0);	
}
string getUserName(int id) {
	string who = ""; 
	for (int i = 0; i < filteredNames.size(); i++) {
		pair<int, string> imgPair = filteredNames.at(i);
		if (imgPair.first == id) {
			who = to_string(imgPair.first) + " " + imgPair.second;
		}
		if (personLookup.at(i).first == id) {
			Person p = peopleImages.at(personLookup.at(i).second).getPerson();
			time_t thetime = p._lastplayed;
			time_t t10minsago  = time(0) - (10 * 60);
			// play welcome sound if more than 10 mins since last time or mnever played it
			if (!p._playedsound || thetime < t10minsago){ 
				if (p._soundfile.size() > 0) {
					peopleImages.at(personLookup.at(i).second).setPlayedSound(true);
					playsound(p._soundfile);
				}
			}
		}
	}
	return who;
}
int main() {
	vector<Mat>  training_set;
	vector<Rect> faces;
	vector<Rect> eyes;
	Mat          m;
	getDBUsers();					// get user list
	getSqlImageDB();				// get image database
	FaceDetect fd(DET_SCALE_FACTOR, DET_MIN_NEIGHBORS,
		DET_MIN_SIZE_RATIO, DET_MAX_SIZE_RATIO, DET_WIDTH, DET_HEIGHT);
	PersonRecognise pr(images, filteredIds, LBPH_RADIUS, LBPH_NEIGHBORS,
		DET_WIDTH, DET_HEIGHT, LBPH_GRID_X, LBPH_GRID_Y, LBPH_THRESHOLD);
	PersonRecognise praff(objimages, objfilteredIds, LBPH_RADIUS, LBPH_NEIGHBORS,
		DET_WIDTH, DET_HEIGHT, LBPH_GRID_X, LBPH_GRID_Y, LBPH_THRESHOLD, PersonType::RAFF);
	VideoCapture cap;				//initialize capture
	cap.open(0);
	namedWindow("window", 1);		//create window to show image
	while (1) {
		cap >> m;				//copy webcam stream to image
		if (!m.empty()) {
			fd.findFacesInImage(m, faces);
			bool has_match = false;
			double match_conf = 0;
			string whoisit = "";

			//analyze each detected face:
			for (vector<Rect>::const_iterator face = faces.begin(); face != faces.end(); face++) {
				Scalar color = NO_MATCH_COLOR;
				Mat face_img = m(*face);
				int label = -1;
				double confidence = 0;
				bool face_match = false;
				//try to recognize the face:
				if (pr.recognize(face_img, label, confidence)) {
					color = MATCH_COLOR;
					has_match = true;
					face_match = true;
					whoisit = getUserName(label);
					// TODO: send an alert to say I'm home
				}
				match_conf = confidence;

				Point center(face->x + face->width * 0.5, face->y + face->height * 0.5);
				circle(m, center, FACE_RADIUS_RATIO * face->width, color, CIRCLE_THICKNESS, LINE_TYPE, 0);
				fd.findEyes(face_img, eyes);
				for (vector<Rect>::const_iterator eye = eyes.begin(); eye != eyes.end(); eye++) {
					Point center(eye->x + eye->width * 0.5, eye->y + eye->height * 0.5);
					circle(face_img, center, EYE_RADIUS_RATIO * eye->width, EYE_COLOR, 
						CIRCLE_THICKNESS_EYE, LINE_TYPE, 0);
				}
			}
/*			Mat_<Vec4f> pos;
			Mat objtest = fd.normalizeImage(m, true);
			if (praff.detectObject(objtest, pos)) {
				for (int i = 0; i < pos.total(); i++)
				{
					Vec4f p = pos.at<Vec4f>(i);
					circle(m, Point(p[0], p[1]), DET_HEIGHT / 2, Scalar(255), 2);
				}
			}
*/
			putText(m, format("FPS: %d", 15), cvPoint(10, m.rows - 80),
				FONT, 2, FONT_COLOR, 1, LINE_TYPE);
			putText(m, format("Faces: %d", faces.size()), cvPoint(10, m.rows - 55),
				FONT, 2, FONT_COLOR, 1, LINE_TYPE);
			putText(m, format("Match: %s", whoisit), cvPoint(10, m.rows - 30),
				FONT, 2, FONT_COLOR, 1, LINE_TYPE);
			putText(m, format("Confidence: %f", match_conf), cvPoint(10, m.rows - 5),
				FONT, 2, FONT_COLOR, 1, LINE_TYPE);

			imshow("window", m);	//print image to screen
		}
		waitKey(33);				//delay 33ms
	}
	return 0;
}


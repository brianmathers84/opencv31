#ifndef DEFS_H
#define	DEFS_H

/** Flags: **/
#define SHOW_OUTPUT
#define WRITE_OUTPUT
#define WRITE_CSV

/** Inputs: **/
#define CASCADE_PATH	"cascades/haarcascade_frontalface_default.xml"
#define FACE_DETECT		"cascades/haarcascade_frontalface_default.xml"
#define EYE_DETECT		"cascades/haarcascade_eye.xml"

/** Colors, fonts, lines... **/
#define NO_MATCH_COLOR    Scalar(0,0,255) //red
#define MATCH_COLOR       Scalar(0,255,0) //green
#define EYE_COLOR		  Scalar(0,255,255) //blue
#define FACE_RADIUS_RATIO 0.6
#define EYE_RADIUS_RATIO 0.4
#define CIRCLE_THICKNESS  2.0
#define CIRCLE_THICKNESS_EYE  0.5
#define LINE_TYPE         CV_AA
#define FONT              FONT_HERSHEY_PLAIN
#define FONT_COLOR        Scalar(0,255,255)
#define THICKNESS_TITLE   1.9
#define SCALE_TITLE       1.9
#define POS_TITLE         cvPoint(10, 30)
#define THICKNESS_LINK    1.6
#define SCALE_LINK        1.3
#define POS_LINK          cvPoint(10, 55)

/** Face detector: **/
#define DET_SCALE_FACTOR   1.05
#define DET_MIN_NEIGHBORS  40
#define DET_MIN_SIZE_RATIO 0.06
#define DET_MAX_SIZE_RATIO 3.5
#define DET_HEIGHT 200
#define DET_WIDTH 150
#define DET_MATCH_COLOR CV_LOAD_IMAGE_GRAYSCALE;

/** LBPH face recognizer: **/
#define LBPH_RADIUS    3
#define LBPH_NEIGHBORS 8
#define LBPH_GRID_X    8
#define LBPH_GRID_Y    8
#define LBPH_THRESHOLD 180.0

#endif	/* DEFS_H */


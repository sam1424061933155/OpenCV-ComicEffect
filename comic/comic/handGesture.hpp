//
//  handGesture.hpp
//  comicingimage
//
//  Created by 張修齊 on 2017/6/13.
//  Copyright © 2017年 張修齊. All rights reserved.
//

#ifndef _HAND_GESTURE_
#define _HAND_GESTURE_

#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "main.hpp"
#include "myImage.hpp"

using namespace cv;
using namespace std;

class HandGesture{
public:
    MyImage m;
    HandGesture();
    vector<vector<Point> > contours;
    vector<vector<int> >hullI;
    vector<vector<Point> >hullP;
    vector<vector<Vec4i> > defects;
    vector <Point> fingerTips;
    Rect rect;
    void printGestureInfo(Mat src);
    int cIdx;
    int frameNumber;
    int mostFrequentFingerNumber;
    int nrOfDefects;
    Rect bRect;
    double bRect_width;
    double bRect_height;
    bool isHand;
    bool detectIfHand();
    void initVectors();
    void getFingerNumber(MyImage *m);
    void eleminateDefects(MyImage *m);
    void getFingerTips(MyImage *m);
    void drawFingerTips(MyImage *m);
private:
    string bool2string(bool tf);
    int fontFace;
    int prevNrFingerTips;
    void checkForOneFinger(MyImage *m);
    float getAngle(Point s,Point f,Point e);
    vector<int> fingerNumbers;
    void analyzeContours();
    string intToString(int number);
    void computeFingerNumber();
    void drawNewNumber(MyImage *m);
    void addNumberToImg(MyImage *m);
    vector<int> numbers2Display;
    void addFingerNumberToVector();
    Scalar numberColor;
    int nrNoFinger;
    float distanceP2P(Point a,Point b);
    void removeRedundantEndPoints(vector<Vec4i> newDefects,MyImage *m);
    void removeRedundantFingerTips();
};




#endif

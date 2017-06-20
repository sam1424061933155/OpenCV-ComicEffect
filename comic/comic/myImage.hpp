//
//  myImage.hpp
//  comicingimage
//
//  Created by 張修齊 on 2017/6/13.
//  Copyright © 2017年 張修齊. All rights reserved.
//

#ifndef _MYIMAGE_
#define _MYIMAGE_

#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class MyImage{
public:
    MyImage(int webCamera);
    MyImage();
    Mat srcLR;
    Mat src;
    Mat bw;
    vector<Mat> bwList;
    VideoCapture cap;
    int cameraSrc;
    void initWebCamera(int i);
};



#endif

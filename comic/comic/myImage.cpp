//
//  myImage.cpp
//  comicingimage
//
//  Created by 張修齊 on 2017/6/13.
//  Copyright © 2017年 張修齊. All rights reserved.
//

#include "myImage.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

using namespace cv;

MyImage::MyImage(){
}

MyImage::MyImage(int webCamera){
    cameraSrc=webCamera;
    cap=VideoCapture(webCamera);
}


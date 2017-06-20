//
//  cartoon.cpp
//  comic
//
//  Created by sam on 2017/6/8.
//  Copyright © 2017年 sam. All rights reserved.
//

#include "cartoon.h"
#include <opencv2/highgui/highgui.hpp> // For VS2015
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

void cartoonTransform(Mat &Frame,Mat &output){
    
    Mat grayImage;
    cvtColor(Frame, grayImage, CV_BGR2GRAY);
    medianBlur(grayImage, grayImage, 7);
    Mat edge;
    Laplacian(grayImage, edge, CV_8U,5);
    Mat Binaryzation;
    threshold(edge, Binaryzation, 80, 255, THRESH_BINARY_INV);
    Size size = Frame.size();
    Size reduceSize;
    reduceSize.width = size.width/2;
    reduceSize.height = size.height/2;
    Mat reduceImage = Mat(reduceSize, CV_8UC3);
    resize(Frame, reduceImage, reduceSize);
    Mat tmp = Mat(reduceSize, CV_8UC3);
    int repetitions = 7;
    for (int i=0 ; i < repetitions; i++)
    {
        int kernelSize = 9;
        double sigmaColor = 9;
        double sigmaSpace = 7;
        bilateralFilter(reduceImage, tmp, kernelSize, sigmaColor, sigmaSpace);
        bilateralFilter(tmp, reduceImage, kernelSize, sigmaColor, sigmaSpace);
    }
    
    Mat magnifyImage;
    resize(reduceImage, magnifyImage, size);
    Mat dst;
    dst.setTo(0);
    magnifyImage.copyTo(dst,Binaryzation);
    output = dst;
    //imshow("Carton", output);

    
}

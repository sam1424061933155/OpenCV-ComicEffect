//
//  main.cpp
//  comic
//
//  Created by sam on 2017/6/6.
//  Copyright © 2017年 sam. All rights reserved.
//
#include <opencv2/highgui/highgui.hpp> // For VS2015
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv/cvaux.hpp>
#include <opencv2/video/video.hpp>
#include "cartoon.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

/** Global variables */
String face_cascade_name = "/Users/sam/Desktop/cv_env/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "/Users/sam/Desktop/cv_env/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;



Mat src,dst;
int spatialRad=10,colorRad=10,maxPryLevel=4;


void cartoonTransform(Mat &Frame,Mat &output){
   

    Mat grayImage;
    cvtColor(Frame, grayImage, CV_RGBA2GRAY);
    
    Mat bg_shadow = imread("/Users/sam/Desktop/shadow1.png");
    if(!bg_shadow.data){
        cout<<"not read shadow"<<endl;
    }
    Mat lineMat;
    Mat mask(output.rows,output.cols,2);
    GaussianBlur(grayImage, lineMat, Size(3,3) ,0 ,0);
    
    for( int y = 0; y < grayImage.rows; y++ )
    {
        for( int x = 0; x < grayImage.cols; x++ )
        {
            for( int c = 0; c < 3; c++ )
            {
                if(grayImage.at<Vec3b>(y,x)[c]<70){
                    grayImage.at<Vec3b>(y,x)[c]=0;
                    mask.at<Vec3b>(y,x)[c]=1;
                }else if(70 <= grayImage.at<Vec3b>(y,x)[c] && grayImage.at<Vec3b>(y,x)[c] < 120){
                    grayImage.at<Vec3b>(y,x)[c]=100;
                }else{
                    grayImage.at<Vec3b>(y,x)[c]=255;
                    mask.at<Vec3b>(y,x)[c]=1;
                }
            }
        }
    }

    grayImage.copyTo(output);
    //Canny(lineMat , lineMat, 20, 120, 3);
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Sobel(lineMat, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);  //轉成CV_8U
    Sobel(lineMat, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );
    convertScaleAbs(grad_y, abs_grad_y);
    
    Mat dst1, dst2;
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst1);
    threshold(dst1, lineMat, 80, 255, THRESH_BINARY|THRESH_OTSU);
    imshow("canny", lineMat);
    //Sobel(lineMat, lineMat, CV_8U, 1, 0,3,4,0,BORDER_DEFAULT);
    lineMat.copyTo(mask);
    bitwise_not(lineMat, lineMat);
    lineMat.copyTo(output,mask);
    cvtColor(output, output, CV_GRAY2BGRA);
    imshow("output brfore", output);
    cout <<"output "<< output.cols<<" , "<<output.rows<<endl;
    cout <<"Frame "<< Frame.cols<<" , "<<Frame.rows<<endl;
    
    for( int y = 0; y < output.rows; y++ )
    {
        for( int x = 0; x < output.cols; x++ )
        {
            for( int c = 0; c < 3; c++ )
            {
                output.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( 0.9*( output.at<Vec3b>(y,x)[c] )  );
            }
        }
    }

    for( int y = 0; y < output.rows; y++ )
    {
        for( int x = 0; x < output.cols; x++ )
        {
            for( int c = 0; c < 3; c++ )
            {
                if(output.at<Vec3b>(y,x)[c]>50 && output.at<Vec3b>(y,x)[c]<150){
                    int y_value = y % bg_shadow.rows;
                    int x_value = x % bg_shadow.cols;
                    output.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( bg_shadow.at<Vec3b>(y_value,x_value)[c] )  ;
                }
            }
        }
    }

    
}



int main(int argc, const char * argv[]) {
    
    
    int cameraNumber = 0; // 设定摄像头编号为0
    //cartoon photo;
    
    if(argc > 1)
        cameraNumber = atoi(argv[1]);
    
    // 开启摄像头
    cv::VideoCapture camera;
    camera.open(cameraNumber);
    if(!camera.isOpened())
    {
        cout <<"Error: Could not open the camera."<<endl;
        exit(1);
    }
    
    // 调整摄像头的输出分辨率
    camera.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    
    face_cascade.load(face_cascade_name);
    eyes_cascade.load(eyes_cascade_name);
    
    while (1)
    {
        Mat Frame;
        camera >> Frame;
        if(!Frame.data)
        {
            cout << "Couldn't capture camera frame.";
            exit(1);
        }
        
        // 创建一个用于存放输出图像的数据结构
        Mat output(Frame.size(), CV_8UC3);
        cartoonTransform(Frame, output);
        
        // 使用图像处理技术将获取的帧经过处理后输入到output中
        //imshow("Original", Frame);
        namedWindow("Cartoon", CV_WINDOW_NORMAL);

        imshow("Cartoon", output);
        
        char keypress = waitKey(20);
        
        if(keypress == 27)
        {
            break;
        }
        
    }
    
    
    return 0;
}

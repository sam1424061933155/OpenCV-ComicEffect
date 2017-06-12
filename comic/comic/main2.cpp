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

using namespace std;
using namespace cv;

/** Global variables */
String face_cascade_name = "/Users/sam/Desktop/cv_env/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "/Users/sam/Desktop/cv_env/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;






void cartoonTransform(Mat &Frame,Mat &output){
   
    
    Mat grayImage;
    cvtColor(Frame, grayImage, CV_BGR2GRAY);
    medianBlur(grayImage, grayImage, 7);
    Mat edge;
    //Laplacian(grayImage, edge, CV_8U,5);
    Sobel(grayImage, edge, CV_8U, 1, 0,3,5,0,BORDER_DEFAULT);
    //dilate(edge, edge, Mat());
    Mat Binaryzation;
    threshold(edge, Binaryzation, 128, 255, THRESH_BINARY_INV);
    Size size = Frame.size();
    Size reduceSize;
    reduceSize.width = size.width/2;
    reduceSize.height = size.height/2;
    Mat reduceImage = Mat(reduceSize, CV_8UC3);
    resize(Frame, reduceImage, reduceSize);
    Mat tmp = Mat(reduceSize, CV_8UC3);
    int repetitions = 0;
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
    
    float alpha=1.5;
    float beta= 30;
    
    for( int y = 0; y < Frame.rows; y++ )
    {
        for( int x = 0; x < Frame.cols; x++ )
        {
            for( int c = 0; c < 3; c++ )
            {
                Frame.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( alpha*( Frame.at<Vec3b>(y,x)[c] ) + beta );
            }
        }
    }
    //cvtColor(output, output, CV_BGR2GRAY);
    threshold(output, output, 85, 255, CV_THRESH_BINARY);
    
    Point ball;
   
    std::vector<Rect> faces;
    Mat frame_gray;
    
    cvtColor(Frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(1, 1) );
    
    for( int i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        ellipse( output, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 0, 255, 0 ), 2, 8, 0 );
        
        Mat faceROI = frame_gray( faces[i] );
        std::vector<Rect> eyes;
        
        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(.5, .5) );
        
        for( int j = 0; j < eyes.size(); j++ )
        {
            Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
            int radius = cvRound( (eyes[j].width + eyes[i].height)*0.25 );
            circle( output, center, radius, Scalar( 255, 0, 0 ), 3, 8, 0 );
            ball = center;
        }
    }
    Mat glass = imread("/Users/sam/Desktop/mut.png");
    if(!glass.data){
        cout<<"not read logo"<<endl;
    }
    //cout << "col "<<output.cols<<" row "<<output.rows<<endl;
    //cout << "col "<<glass.cols<<" row "<<glass.rows<<endl;
    Mat imageROI=output(Rect(ball.x,ball.y,glass.cols,glass.rows)); //獲取感興趣區域，即logo要放置的區域
    Mat mask= imread("/Users/sam/Desktop/mut.png",0);
    glass.copyTo(imageROI,mask);
   
    
    
    
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
    camera.set(CV_CAP_PROP_FRAME_WIDTH, 1000);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT, 1000);
    
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
        imshow("Carton", output);
        
        char keypress = waitKey(20);
        
        if(keypress == 27)
        {
            break;
        }
        
    }
    
    
    return 0;
}

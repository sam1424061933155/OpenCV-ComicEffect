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
String eyes_cascade_name = "/Users/sam/Desktop/cv_env/opencv_contrib/modules/face/data/cascades/haarcascade_mcs_eyepair_small.xml";
String right_eyes_cascade_name = "/Users/sam/Desktop/cv_env/opencv_contrib/modules/face/data/cascades/haarcascade_mcs_righteye.xml";
String left_eyes_cascade_name = "/Users/sam/Desktop/cv_env/opencv_contrib/modules/face/data/cascades/haarcascade_mcs_lefteye.xml";

String nose_cascade_name = "/Users/sam/Desktop/cv_env/opencv_contrib/modules/face/data/cascades/haarcascade_mcs_nose.xml";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier right_eyes_cascade;
CascadeClassifier left_eyes_cascade;

CascadeClassifier nose_cascade;

char keypress=49;





void cartoonTransform(Mat &Frame,Mat &output,int type){

    Mat grayImage;
    cvtColor(Frame, grayImage, CV_BGR2GRAY);  //CV_RGBA2GRAY
    
    Mat bg_shadow = imread("/Users/sam/Desktop/course/cv/final_project/shadow1.png");
    //imshow("shadow",bg_shadow);

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
    //imshow("canny", lineMat);
    //Sobel(lineMat, lineMat, CV_8U, 1, 0,3,4,0,BORDER_DEFAULT);
    lineMat.copyTo(mask);
    bitwise_not(lineMat, lineMat);
    lineMat.copyTo(output,mask);
    cvtColor(output, output, CV_GRAY2BGR);  //CV_GRAY2BGRA
    imshow("output brfore", output);
    //cout <<"output "<< output.cols<<" , "<<output.rows<<endl;
    //cout <<"Frame "<< Frame.cols<<" , "<<Frame.rows<<endl;
    
    //變暗
    
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
    
    //add shadow

    for( int y = 0; y < output.rows; y++ )
    {
        for( int x = 0; x < output.cols; x++ )
        {
            for( int c = 0; c < 4; c++ )
            {
                if(output.at<Vec3b>(y,x)[c]>89 && output.at<Vec3b>(y,x)[c]<150){ //85-90
                    int y_value = y % bg_shadow.rows;
                    int x_value = x % bg_shadow.cols;
                    output.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( bg_shadow.at<Vec3b>(y_value,x_value)[c] )  ;
                }
            }
        }
    }
    
    
    //-- Detect faces
    double mut_pos_x=0;
    double mut_pos_y=0;
    double right_eye_pos_x=0;
    double right_eye_pos_y=0;
    double left_eye_pos_x=0;
    double left_eye_pos_y=0;

    Point pos_start,pos_end;
        
    vector<Rect> faces;
    Mat frame_gray;
        
    cvtColor(Frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
        
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(1, 1) );
        
    for( int i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        //ellipse( output, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 0, 255, 0 ), 2, 8, 0 );
        pos_start.x = center.x - faces[i].width*0.5;
        pos_end.x = center.x + faces[i].width*0.5;
            
        pos_start.y = center.y - faces[i].height*0.5;
        pos_end.y = center.y + faces[i].height*0.5;
        
        
            
        Mat faceROI = frame_gray( faces[i] );
        vector<Rect> right_eyes,left_eyes,eyes;
        Point eye_center;
        int eye_radius;
            
        //-- In each face, detect eyes
        
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(.5, .5) );
        cout <<"eye "<<eyes.size()<<endl;
        for( int j = 0; j < eyes.size(); j++ )
        {
            eye_center.x = faces[i].x + eyes[j].x + eyes[j].width*0.5;
            eye_center.y = faces[i].y + eyes[j].y + eyes[j].height*0.5;
            eye_radius = cvRound( (eyes[j].width + eyes[i].height)*0.25 );
            //cout <<"radius "<<eye_radius<<endl;
            circle( output, eye_center, eye_radius, Scalar( 0 , 0, 255), 3, 8, 0 );
            if(eye_radius > 50){
                right_eye_pos_x = eye_center.x+eye_radius*0.55;
                right_eye_pos_y = eye_center.y-eye_radius*0.8;
                left_eye_pos_x = eye_center.x-eye_radius*1.65;
                left_eye_pos_y = eye_center.y-eye_radius*0.8;
            }
            
            
        }

        
        /*right_eyes_cascade.detectMultiScale( faceROI, right_eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(.5, .5) );
        for( int j = 0; j < right_eyes.size(); j++ )
        {
            Point center( faces[i].x + right_eyes[j].x + right_eyes[j].width*0.5, faces[i].y + right_eyes[j].y + right_eyes[j].height*0.5 );
            int radius = cvRound( (right_eyes[j].width + right_eyes[i].height)*0.25 );
            //circle( output, center, radius, Scalar( 255, 0, 0 ), 3, 8, 0 );
        }
        left_eyes_cascade.detectMultiScale( faceROI, left_eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(.5, .5) );
        for( int j = 0; j < left_eyes.size(); j++ )
        {
            Point center( faces[i].x + left_eyes[j].x + left_eyes[j].width*0.5, faces[i].y + left_eyes[j].y + left_eyes[j].height*0.5 );
            int radius = cvRound( (left_eyes[j].width + left_eyes[i].height)*0.25 );
            //circle( output, center, radius, Scalar(0, 255, 0 ), 3, 8, 0 );
            
        }*/
        
        vector<Rect> noses;
        nose_cascade.detectMultiScale(faceROI, noses, 1.1 , 2 , 0 |CV_HAAR_SCALE_IMAGE,Size(0.5,0.5));
        for( int k = 0; k < noses.size(); k++ )
        {
            Point center( faces[i].x + noses[k].x + noses[k].width*0.5, faces[i].y + noses[k].y + noses[k].height*0.5 );
            int radius = cvRound( (noses[k].width + noses[i].height)*0.25 );
            //circle( output, center, radius, Scalar( 255, 0, 0 ), 3, 8, 0 );
            //pos = center;
            mut_pos_x = center.x;
            mut_pos_y = center.y;
        }
            
    }
        
        
        
        
    if(type==2){
        cout <<" in type 2 clause"<<endl;

        Mat mut = imread("/Users/sam/Desktop/course/cv/final_project/mut.png",CV_LOAD_IMAGE_COLOR);
        
        if(!mut.data){
            cout<<"not read mut"<<endl;
        }
        Mat mut_gray;
        mut_gray = mut;
        
        Size reduceSize;
        
        reduceSize.width = 128;
        reduceSize.height = 128;
        resize(mut_gray, mut_gray, reduceSize);
        
        //imshow("mut",mut_gray);
        
        mut_pos_x = mut_pos_x - mut_gray.cols*0.5;
        mut_pos_y = mut_pos_y - mut_gray.rows*0.2;
        
        
        //cout <<"mut pos x and y "<<endl;
        //cout <<mut_pos_x<<" , "<<mut_pos_y<<endl;
        
        //cout <<"mut_gray pos x and y "<<endl;
        //cout <<mut_pos_x+mut_gray.cols<<" , "<<mut_pos_y+mut_gray.rows<<endl;
        
        
        if(pos_start.x<=mut_pos_x+mut_gray.cols && pos_end.x>=mut_pos_x+mut_gray.cols && pos_start.y<=mut_pos_y+mut_gray.rows && pos_end.y>=mut_pos_y+mut_gray.rows ){
            Mat mask = imread("/Users/sam/Desktop/course/cv/final_project/mut.png",0); //注意要是灰度图才行
            resize(mask, mask, reduceSize);
            
            threshold(mask,mask,120,255,CV_THRESH_BINARY);
            Mat mask1 = mask; //掩模反色
            
            //imshow("mask1", mask1);
            
            Mat imageROI=output(Rect(mut_pos_x,mut_pos_y,mut_gray.cols,mut_gray.rows)); //獲取感興趣區域，即logo要放置的區域  //size問題
            //cout<<"pos "<<pos.x<<" , "<<pos.y<<endl;
            //cout<<"mut_gray "<<mut_gray.cols<<" , "<<mut_gray.rows<<" , "<<mut_gray.channels()<<endl;
            
            //cout<<"output "<<output.cols<<" , "<<output.rows<<" , "<<output.channels()<<endl;
            //cout<<"imageROI "<<imageROI.cols<<" , "<<imageROI.rows<<" , "<<imageROI.channels()<<endl;
            //addWeighted(imageROI,0.5,mut_gray,0.5,0,imageROI);
            mut_gray.copyTo(imageROI,mask1);
        }
        
    }else if(type==3){
        cout <<" in type 3 clause"<<endl;
        Mat eye = imread("/Users/sam/Desktop/course/cv/final_project/eye.png",CV_LOAD_IMAGE_COLOR);
        
        if(!eye.data){
            cout<<"not read eye"<<endl;
        }
        //imshow("bat", water);
        Mat eye_gray;
        eye_gray = eye;
        Size reduceSize;
        
        reduceSize.width = 80;
        reduceSize.height = 80;

        
        resize(eye_gray, eye_gray,reduceSize);
        
        //imshow("mut",mut_gray);
        
       // mut_pos_x = mut_pos_x - mut_gray.cols*0.5;
        //mut_pos_y = mut_pos_y - mut_gray.rows*0.2;
    
        
        if(pos_start.x<=right_eye_pos_x+eye_gray.cols && pos_end.x>=right_eye_pos_x+eye_gray.cols && pos_start.y<=right_eye_pos_y+eye_gray.rows && pos_end.y>=right_eye_pos_y+eye_gray.rows ){
            Mat mask = imread("/Users/sam/Desktop/course/cv/final_project/eye.png",0); //注意要是灰度图才行
            resize(mask, mask, reduceSize);
            threshold(mask,mask,120,255,CV_THRESH_BINARY);
            //imshow("mask",mask);
            Mat mask1 = 1-mask; //掩模反色
            Mat imageROI=output(Rect(right_eye_pos_x,right_eye_pos_y,eye_gray.cols,eye_gray.rows)); //獲取感興趣區域，即logo要放置的區域  //size問題
            eye_gray.copyTo(imageROI,mask1);
            Mat imageROI1=output(Rect(left_eye_pos_x,left_eye_pos_y,eye_gray.cols,eye_gray.rows)); //獲取感興趣區域，即logo要放置的區域  //size問題
            eye_gray.copyTo(imageROI1,mask1);
        }
        

    }else if(type ==4){
        cout <<"in type4 if clause"<<endl;
        
        
    }


    
}
void sketch(Mat &Frame,Mat &output){
   /* GaussianBlur(Frame, Frame, Size(3,3) ,0 ,0);
    Mat grayImage;
    cvtColor(Frame, grayImage, CV_BGR2GRAY);  //CV_RGBA2GRAY
    float threshold_value =10;
    Mat dst(Frame.size(), CV_8UC1);
    for(int i=0;i<grayImage.rows;i++){
        for(int j=0;j<grayImage.cols;j++){
            if(i!=grayImage.cols-1 && j!=grayImage.rows-1){
                float src_value = grayImage.at<uchar>(i,j);
                float dst_value = grayImage.at<uchar>(i+1,j+1);
                float diff = abs(src_value-dst_value);
                if(diff > threshold_value){
                    dst.at<float>(i,j)=0;
                }else{
                    dst.at<float>(i,j)=255;
                }
                
            }
        }
    }
    for( int i = 0; i < grayImage.rows; i++ )
    {
        for( int j = 0; j < grayImage.cols; j++ )
        {
            for( int c = 0; c < 1; c++ )
            {
                if(i!=grayImage.cols-1 && j!=grayImage.rows-1){
                    float src_value = saturate_cast<uchar>( grayImage.at<Vec3b>(i,j)[c]);
                    float dst_value = saturate_cast<uchar>( grayImage.at<Vec3b>(i+1,j+1)[c]);
                    float diff = abs(src_value-dst_value);
                    if(diff > threshold_value){
                        dst.at<Vec3b>(i,j)[c]=0;
                    }else{
                        dst.at<Vec3b>(i,j)[c]=255;
                    }
                    
                }
            }
        }
    }
    
    output = dst;*/
    
    Mat image2 = Frame.clone();
    cvtColor(Frame, Frame, CV_BGR2GRAY);//灰度图
    Mat sobel_x, sobel_y;
    Sobel(Frame, sobel_x, CV_16S, 1, 0);  // Sobel
    Sobel(Frame, sobel_y, CV_16S, 0, 1);
    Mat sobel;
    sobel = abs(sobel_x) + abs(sobel_y);
    
    double sobmin, sobmax;
    minMaxLoc(sobel, &sobmin, &sobmax);
    Mat sobelImage;
    sobel.convertTo(sobelImage, CV_8U, -255.0 / sobmax, 255);
    
    int mul = 2;
    //resize(sobelImage, sobelImage, Size(640/mul, 480/mul));
    
    for (int i = 0; i < sobelImage.rows; ++i) {
        for (int j = 0; j < sobelImage.cols; ++j) {
            image2.data[3*(j + i*image2.cols+image2.cols-sobelImage.cols)] = sobelImage.data[j + i*sobelImage.cols];
            image2.data[3 * (j + i*image2.cols + image2.cols - sobelImage.cols)+1] = sobelImage.data[j + i*sobelImage.cols];
            image2.data[3 * (j + i*image2.cols + image2.cols - sobelImage.cols)+2] = sobelImage.data[j + i*sobelImage.cols];
        }
        
    }
    output = sobelImage;
    
    

}
void cartoon(Mat &Frame,Mat &output){
    Mat grayImage;
    cvtColor(Frame, grayImage, CV_BGR2GRAY);
    medianBlur(grayImage, grayImage, 7);
    Mat edge;
    //Laplacian(grayImage, edge, CV_8U,5);
    Sobel(grayImage, edge, CV_8U, 1, 0,3,4,0,BORDER_DEFAULT);
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
    right_eyes_cascade.load(right_eyes_cascade_name);
    left_eyes_cascade.load(left_eyes_cascade_name);
    nose_cascade.load(nose_cascade_name);
    
    int type=1;

    
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
       
        
        char keypress = waitKey(20);
        
        if(keypress == 27)
        {
            break;
        }
        
        if(keypress == 49){
            cout <<"1 is click "<<endl;
            type =1;
        }else if(keypress == 50){
            cout <<"2 is click "<<endl;
            type=2;
        }else if(keypress == 51){
            cout <<"3 is click "<<endl;
            type=3;
        }else if(keypress == 52){
            cout <<"4 is click "<<endl;
            type=4;

        }else if(keypress == 53){
            cout <<"5 is click "<<endl;
            type=5;
            
        }
        if(type!=4 && type!=5){
            cartoonTransform(Frame, output,type);

        }else if(type==4){
            cartoon(Frame, output);
        }else{
            sketch(Frame, output);
        }
        //sketch(Frame,output);
        // 使用图像处理技术将获取的帧经过处理后输入到output中
        namedWindow("Cartoon", CV_WINDOW_NORMAL);
        imshow("Cartoon", output);
        
    }
    
    
    return 0;
}

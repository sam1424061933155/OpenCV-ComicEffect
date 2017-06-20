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
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image_abstract.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/highgui/highgui_c.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include "myImage.hpp"
#include "roi.hpp"
#include "handGesture.hpp"
#include <vector>
#include <cmath>
#include "main.hpp"

using namespace std;
using namespace cv;

struct correspondens{
    vector<int> index;
};

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



/* Global Variables  */
int fontFace = FONT_HERSHEY_PLAIN;
int square_len;
int avgColor[NSAMPLES][3] ;
int c_lower[NSAMPLES][3];
int c_upper[NSAMPLES][3];
int avgBGR[3];
int nrOfDefects;
int iSinceKFInit;
struct dim{int w; int h;}boundingDim;
VideoWriter out;
Mat edges;
My_ROI roi1, roi2,roi3,roi4,roi5,roi6;
vector <My_ROI> roi;
vector <KalmanFilter> kf;
vector <Mat_<float> > measurement;

/* end global variables */

void init(MyImage *m){
    square_len=20;
    iSinceKFInit=0;
}

// change a color from one space to another
void col2origCol(int hsv[3], int bgr[3], Mat src){
    Mat avgBGRMat=src.clone();
    for(int i=0;i<3;i++){
        avgBGRMat.data[i]=hsv[i];
    }
    cvtColor(avgBGRMat,avgBGRMat,COL2ORIGCOL);
    for(int i=0;i<3;i++){
        bgr[i]=avgBGRMat.data[i];
    }
}

void printText(Mat src, string text){
    int fontFace = FONT_HERSHEY_PLAIN;
    putText(src,text,Point(src.cols/2, src.rows/10),fontFace, 1.2f,Scalar(200,0,0),2);
}

void waitForPalmCover(MyImage* m){
    m->cap >> m->src;
    flip(m->src,m->src,1);
    roi.push_back(My_ROI(Point(m->src.cols/3, m->src.rows/6),Point(m->src.cols/3+square_len,m->src.rows/6+square_len),m->src));
    roi.push_back(My_ROI(Point(m->src.cols/4, m->src.rows/2),Point(m->src.cols/4+square_len,m->src.rows/2+square_len),m->src));
    roi.push_back(My_ROI(Point(m->src.cols/3, m->src.rows/1.5),Point(m->src.cols/3+square_len,m->src.rows/1.5+square_len),m->src));
    roi.push_back(My_ROI(Point(m->src.cols/2, m->src.rows/2),Point(m->src.cols/2+square_len,m->src.rows/2+square_len),m->src));
    roi.push_back(My_ROI(Point(m->src.cols/2.5, m->src.rows/2.5),Point(m->src.cols/2.5+square_len,m->src.rows/2.5+square_len),m->src));
    roi.push_back(My_ROI(Point(m->src.cols/2, m->src.rows/1.5),Point(m->src.cols/2+square_len,m->src.rows/1.5+square_len),m->src));
    roi.push_back(My_ROI(Point(m->src.cols/2.5, m->src.rows/1.8),Point(m->src.cols/2.5+square_len,m->src.rows/1.8+square_len),m->src));
    
    
    for(int i =0;i<50;i++){
        m->cap >> m->src;
        flip(m->src,m->src,1);
        for(int j=0;j<NSAMPLES;j++){
            roi[j].draw_rectangle(m->src);
        }
        string imgText=string("Cover rectangles with palm");
        printText(m->src,imgText);
        
        if(i==30){
            //	imwrite("./images/waitforpalm1.jpg",m->src);
        }
        
        imshow("img1", m->src);
        out << m->src;
        if(cv::waitKey(30) >= 0) break;
    }
}

int getMedian(vector<int> val){
    int median;
    size_t size = val.size();
    sort(val.begin(), val.end());
    if (size  % 2 == 0)  {
        median = val[size / 2 - 1] ;
    } else{
        median = val[size / 2];
    }
    return median;
}


void getAvgColor(MyImage *m,My_ROI roi,int avg[3]){
    Mat r;
    roi.roi_ptr.copyTo(r);
    vector<int>hm;
    vector<int>sm;
    vector<int>lm;
    // generate vectors
    for(int i=2; i<r.rows-2; i++){
        for(int j=2; j<r.cols-2; j++){
            hm.push_back(r.data[r.channels()*(r.cols*i + j) + 0]) ;
            sm.push_back(r.data[r.channels()*(r.cols*i + j) + 1]) ;
            lm.push_back(r.data[r.channels()*(r.cols*i + j) + 2]) ;
        }
    }
    avg[0]=getMedian(hm);
    avg[1]=getMedian(sm);
    avg[2]=getMedian(lm);
}

void average(MyImage *m){
    m->cap >> m->src;
    flip(m->src,m->src,1);
    for(int i=0;i<30;i++){
        m->cap >> m->src;
        flip(m->src,m->src,1);
        cvtColor(m->src,m->src,ORIGCOL2COL);
        for(int j=0;j<NSAMPLES;j++){
            getAvgColor(m,roi[j],avgColor[j]);
            roi[j].draw_rectangle(m->src);
        }
        cvtColor(m->src,m->src,COL2ORIGCOL);
        string imgText=string("Finding average color of hand");
        printText(m->src,imgText);
        imshow("img1", m->src);
        if(cv::waitKey(30) >= 0) break;
    }
}

void initTrackbars(){
    for(int i=0;i<NSAMPLES;i++){
        c_lower[i][0]=12;
        c_upper[i][0]=7;
        c_lower[i][1]=30;
        c_upper[i][1]=40;
        c_lower[i][2]=80;
        c_upper[i][2]=80;
    }
    createTrackbar("lower1","trackbars",&c_lower[0][0],255);
    createTrackbar("lower2","trackbars",&c_lower[0][1],255);
    createTrackbar("lower3","trackbars",&c_lower[0][2],255);
    createTrackbar("upper1","trackbars",&c_upper[0][0],255);
    createTrackbar("upper2","trackbars",&c_upper[0][1],255);
    createTrackbar("upper3","trackbars",&c_upper[0][2],255);
}


void normalizeColors(MyImage * myImage){
    // copy all boundries read from trackbar
    // to all of the different boundries
    for(int i=1;i<NSAMPLES;i++){
        for(int j=0;j<3;j++){
            c_lower[i][j]=c_lower[0][j];
            c_upper[i][j]=c_upper[0][j];
        }
    }
    // normalize all boundries so that
    // threshold is whithin 0-255
    for(int i=0;i<NSAMPLES;i++){
        if((avgColor[i][0]-c_lower[i][0]) <0){
            c_lower[i][0] = avgColor[i][0] ;
        }if((avgColor[i][1]-c_lower[i][1]) <0){
            c_lower[i][1] = avgColor[i][1] ;
        }if((avgColor[i][2]-c_lower[i][2]) <0){
            c_lower[i][2] = avgColor[i][2] ;
        }if((avgColor[i][0]+c_upper[i][0]) >255){
            c_upper[i][0] = 255-avgColor[i][0] ;
        }if((avgColor[i][1]+c_upper[i][1]) >255){
            c_upper[i][1] = 255-avgColor[i][1] ;
        }if((avgColor[i][2]+c_upper[i][2]) >255){
            c_upper[i][2] = 255-avgColor[i][2] ;
        }
    }
}

void produceBinaries(MyImage *m){
    Scalar lowerBound;
    Scalar upperBound;
    Mat foo;
    for(int i=0;i<NSAMPLES;i++){
        normalizeColors(m);
        lowerBound=Scalar( avgColor[i][0] - c_lower[i][0] , avgColor[i][1] - c_lower[i][1], avgColor[i][2] - c_lower[i][2] );
        upperBound=Scalar( avgColor[i][0] + c_upper[i][0] , avgColor[i][1] + c_upper[i][1], avgColor[i][2] + c_upper[i][2] );
        m->bwList.push_back(Mat(m->srcLR.rows,m->srcLR.cols,CV_8U));
        inRange(m->srcLR,lowerBound,upperBound,m->bwList[i]);
    }
    m->bwList[0].copyTo(m->bw);
    for(int i=1;i<NSAMPLES;i++){
        m->bw+=m->bwList[i];
    }
    medianBlur(m->bw, m->bw,7);
}

void initWindows(MyImage m){
    namedWindow("trackbars",CV_WINDOW_KEEPRATIO);
    namedWindow("img1",CV_WINDOW_FULLSCREEN);
}

void showWindows(MyImage m){
    pyrDown(m.bw,m.bw);
    pyrDown(m.bw,m.bw);
    Rect roi( Point( 3*m.src.cols/4,0 ), m.bw.size());
    vector<Mat> channels;
    Mat result;
    for(int i=0;i<3;i++)
        channels.push_back(m.bw);
    merge(channels,result);
    result.copyTo( m.src(roi));
    imshow("img1",m.src);
}

int findBiggestContour(vector<vector<Point> > contours){
    int indexOfBiggestContour = -1;
    int sizeOfBiggestContour = 0;
    for (int i = 0; i < contours.size(); i++){
        if(contours[i].size() > sizeOfBiggestContour){
            sizeOfBiggestContour = contours[i].size();
            indexOfBiggestContour = i;
        }
    }
    return indexOfBiggestContour;
}

void myDrawContours(MyImage *m,HandGesture *hg){
    drawContours(m->src,hg->hullP,hg->cIdx,cv::Scalar(200,0,0),2, 8, vector<Vec4i>(), 0, Point());
    
    
    
    
    rectangle(m->src,hg->bRect.tl(),hg->bRect.br(),Scalar(0,0,200));
    vector<Vec4i>::iterator d=hg->defects[hg->cIdx].begin();
    int fontFace = FONT_HERSHEY_PLAIN;
    
    
    vector<Mat> channels;
    Mat result;
    for(int i=0;i<3;i++)
        channels.push_back(m->bw);
    merge(channels,result);
    //	drawContours(result,hg->contours,hg->cIdx,cv::Scalar(0,200,0),6, 8, vector<Vec4i>(), 0, Point());
    drawContours(result,hg->hullP,hg->cIdx,cv::Scalar(0,0,250),10, 8, vector<Vec4i>(), 0, Point());
    
    
    while( d!=hg->defects[hg->cIdx].end() ) {
   	    Vec4i& v=(*d);
        int startidx=v[0]; Point ptStart(hg->contours[hg->cIdx][startidx] );
        int endidx=v[1]; Point ptEnd(hg->contours[hg->cIdx][endidx] );
        int faridx=v[2]; Point ptFar(hg->contours[hg->cIdx][faridx] );
        float depth = v[3] / 256;
        /*
         line( m->src, ptStart, ptFar, Scalar(0,255,0), 1 );
         line( m->src, ptEnd, ptFar, Scalar(0,255,0), 1 );
         circle( m->src, ptFar,   4, Scalar(0,255,0), 2 );
         circle( m->src, ptEnd,   4, Scalar(0,0,255), 2 );
         circle( m->src, ptStart,   4, Scalar(255,0,0), 2 );
         */
        circle( result, ptFar,   9, Scalar(0,205,0), 5 );
        
        
        d++;
        
    }
    //	imwrite("./images/contour_defects_before_eliminate.jpg",result);
    
}

void makeContours(MyImage *m, HandGesture* hg){
    Mat aBw;
    pyrUp(m->bw,m->bw);
    m->bw.copyTo(aBw);
    findContours(aBw,hg->contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    hg->initVectors();
    hg->cIdx=findBiggestContour(hg->contours);
    if(hg->cIdx!=-1){
        //		approxPolyDP( Mat(hg->contours[hg->cIdx]), hg->contours[hg->cIdx], 11, true );
        hg->bRect=boundingRect(Mat(hg->contours[hg->cIdx]));
        convexHull(Mat(hg->contours[hg->cIdx]),hg->hullP[hg->cIdx],false,true);
        convexHull(Mat(hg->contours[hg->cIdx]),hg->hullI[hg->cIdx],false,false);
        approxPolyDP( Mat(hg->hullP[hg->cIdx]), hg->hullP[hg->cIdx], 18, true );
        if(hg->contours[hg->cIdx].size()>3 ){
            convexityDefects(hg->contours[hg->cIdx],hg->hullI[hg->cIdx],hg->defects[hg->cIdx]);
            hg->eleminateDefects(m);
        }
        bool isHand=hg->detectIfHand();
        hg->printGestureInfo(m->src);
        if(isHand){
            hg->getFingerTips(m);
            hg->drawFingerTips(m);
            myDrawContours(m,hg);
        }
    }
}



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
    //imshow("output brfore", output);
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
            //circle( output, eye_center, eye_radius, Scalar( 0 , 0, 255), 3, 8, 0 );
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
void faceLandmarkDetection(dlib::array2d<unsigned char>& img, dlib::shape_predictor sp, std::vector<Point2f>& landmark)
{
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    
    std::vector<dlib::rectangle> dets = detector(img);
    
    
    dlib::full_object_detection shape = sp(img, dets[0]);
    
    for (int i = 0; i < shape.num_parts(); ++i)
    {
        float x=shape.part(i).x();
        float y=shape.part(i).y();
        landmark.push_back(Point2f(x,y));		
    }
    
    
}
void delaunayTriangulation(const std::vector<Point2f>& hull,std::vector<correspondens>& delaunayTri,Rect rect)
{
    
    cv::Subdiv2D subdiv(rect);
    for (int it = 0; it < hull.size(); it++)
        subdiv.insert(hull[it]);
    //cout<<"done subdiv add......"<<endl;
    std::vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
   
    for (size_t i = 0; i < triangleList.size(); ++i)
    {
        
        std::vector<Point2f> pt;
        correspondens ind;
        Vec6f t = triangleList[i];
        pt.push_back( Point2f(t[0], t[1]) );
        pt.push_back( Point2f(t[2], t[3]) );
        pt.push_back( Point2f(t[4], t[5]) );
        
        if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {
            int count = 0;
            for (int j = 0; j < 3; ++j)
                for (size_t k = 0; k < hull.size(); k++)
                    if (abs(pt[j].x - hull[k].x) < 1.0   &&  abs(pt[j].y - hull[k].y) < 1.0)
                    {
                        ind.index.push_back(k);
                        count++;
                    }
            if (count == 3)
                delaunayTri.push_back(ind);
        }
       
    }	
    
    
}
void applyAffineTransform(Mat &warpImage, Mat &src, std::vector<Point2f> &srcTri, std::vector<Point2f> &dstTri)
{
    // Given a pair of triangles, find the affine transform.
    Mat warpMat = getAffineTransform( srcTri, dstTri );
    
    // Apply the Affine Transform just found to the src image
    warpAffine( src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, BORDER_REFLECT_101);
}

void warpTriangle(Mat &img1, Mat &img2, std::vector<Point2f> &t1, std::vector<Point2f> &t2)
{
    
    Rect r1 = boundingRect(t1);
    Rect r2 = boundingRect(t2);
    
    // Offset points by left top corner of the respective rectangles
    std::vector<Point2f> t1Rect, t2Rect;
    std::vector<Point> t2RectInt;
    for(int i = 0; i < 3; i++)
    {
        
        t1Rect.push_back( Point2f( t1[i].x - r1.x, t1[i].y -  r1.y) );
        t2Rect.push_back( Point2f( t2[i].x - r2.x, t2[i].y - r2.y) );
        t2RectInt.push_back( Point(t2[i].x - r2.x, t2[i].y - r2.y) ); // for fillConvexPoly
        
    }
    
    // Get mask by filling triangle
    Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
    fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);
    
    Mat img1Rect;
    img1(r1).copyTo(img1Rect);
    
    Mat img2Rect = Mat::zeros(r2.height, r2.width, img1Rect.type());
    
    applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);
    
    multiply(img2Rect,mask, img2Rect);
    multiply(img2(r2), Scalar(1.0,1.0,1.0) - mask, img2(r2));
    img2(r2) = img2(r2) + img2Rect;
    
}



void faceSwap(Mat &Frame){
    
    //----------------- step 1. load the input two images. ----------------------------------
    dlib::array2d<unsigned char> imgDlib1,imgDlib2;
    imwrite("/Users/sam/Desktop/course/cv/final_project/frame.png",Frame);
    
    dlib::load_image(imgDlib1,"/Users/sam/Desktop/course/cv/final_project/gordan.png");
    dlib::load_image(imgDlib2,"/Users/sam/Desktop/course/cv/final_project/frame.png" );
    
    Mat imgCV1 = imread("/Users/sam/Desktop/course/cv/final_project/gordan.png");
    Mat imgCV2 = imread("/Users/sam/Desktop/course/cv/final_project/frame.png");
    if(!imgCV1.data || !imgCV2.data)
    {
        printf("No image data \n");
    }
    else
        cout<<"image readed by opencv"<<endl;
    //---------------------- step 2. detect face landmarks -----------------------------------
    dlib::shape_predictor sp;
    dlib::deserialize("/Users/sam/Desktop/course/cv/final_project/shape_predictor_68_face_landmarks.dat") >> sp;
    std::vector<Point2f> points1, points2;
    
    faceLandmarkDetection(imgDlib1,sp,points1);
    faceLandmarkDetection(imgDlib2,sp,points2);
    
    //---------------------step 3. find convex hull -------------------------------------------
    Mat imgCV1Warped = imgCV2.clone();
    imgCV1.convertTo(imgCV1, CV_32F);
    imgCV1Warped.convertTo(imgCV1Warped, CV_32F);
    
    std::vector<Point2f> hull1;
    std::vector<Point2f> hull2;
    std::vector<int> hullIndex;
    
    cv::convexHull(points2, hullIndex, false, false);
    
    for(int i = 0; i < hullIndex.size(); i++)
    {
        hull1.push_back(points1[hullIndex[i]]);
        hull2.push_back(points2[hullIndex[i]]);
    }
    //-----------------------step 4. delaunay triangulation -------------------------------------
    std::vector<correspondens> delaunayTri;
    Rect rect(0, 0, imgCV1Warped.cols, imgCV1Warped.rows);
    delaunayTriangulation(hull2,delaunayTri,rect);
    
    for(size_t i=0;i<delaunayTri.size();++i)
    {
        std::vector<Point2f> t1,t2;
        correspondens corpd=delaunayTri[i];
        for(size_t j=0;j<3;++j)
        {
            t1.push_back(hull1[corpd.index[j]]);
            t2.push_back(hull2[corpd.index[j]]);
        }
        
        warpTriangle(imgCV1,imgCV1Warped,t1,t2);			
    }
    //------------------------step 5. clone seamlessly -----------------------------------------------
    
    //calculate mask
    std::vector<Point> hull8U;
    
    for(int i=0; i< hull2.size();++i)
    {
        Point pt(hull2[i].x,hull2[i].y);
        hull8U.push_back(pt);
    }
    
    
    Mat mask = Mat::zeros(imgCV2.rows,imgCV2.cols,imgCV2.depth());
    fillConvexPoly(mask, &hull8U[0], hull8U.size(), Scalar(255,255,255));
    
    
    Rect r = boundingRect(hull2);
    Point center = (r.tl() +r.br()) / 2;
    
    Mat output;
    imgCV1Warped.convertTo(imgCV1Warped, CV_8UC3);
    seamlessClone(imgCV1Warped,imgCV2,mask,center,output,NORMAL_CLONE);
    
    imshow("result",output);

    
    

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
    int count = 0;

    MyImage m(0);
    HandGesture hg;
    init(&m);
    //m.cap >>m.src;
    namedWindow("img1",CV_WINDOW_KEEPRATIO);
    //out.open("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, m.src.size(), true);
    waitForPalmCover(&m);
    average(&m);
    destroyWindow("img1");
    initWindows(m);
    initTrackbars();
    
    while (1)
    {
        Mat Frame;
        camera >> Frame;
        if(!Frame.data)
        {
            cout << "Couldn't capture camera frame.";
            exit(1);
        }
        
        hg.frameNumber++;
        m.cap >> m.src;
        flip(m.src,m.src,1);
        pyrDown(m.src,m.srcLR);
        blur(m.srcLR,m.srcLR,Size(3,3));
        cvtColor(m.srcLR,m.srcLR,ORIGCOL2COL);
        produceBinaries(&m);
        cvtColor(m.srcLR,m.srcLR,COL2ORIGCOL);
        makeContours(&m, &hg);
        hg.getFingerNumber(&m);
        showWindows(m);
        out << m.src;
        
        // 创建一个用于存放输出图像的数据结构
        Mat output(Frame.size(), CV_8UC3);
       
        
        char keypress = waitKey(20);
        
        if(keypress == 27)
        {
            break;
        }
        /*
        if(hg.mostFrequentFingerNumber==1){
            cout <<"1 is click "<<endl;
            type =1;
        }else if(hg.mostFrequentFingerNumber==2){
            cout <<"2 is click "<<endl;
            type =2;
        }else if(hg.mostFrequentFingerNumber==3){
            cout <<"3 is click "<<endl;
            type =3;
        }else if(hg.mostFrequentFingerNumber==4){
            cout <<"4 is click "<<endl;
            type =4;
        }else if(hg.mostFrequentFingerNumber==5){
            cout <<"5 is click "<<endl;
            type =5;
        }else if(hg.mostFrequentFingerNumber==6){
            cout <<"6 is click "<<endl;
            type =6;
        }
        */
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
            
        }else if(keypress == 54){
            cout <<"6 is click "<<endl;
            type=6;
            
        }
        if(type!=4 && type!=5 && type!=6){
            cartoonTransform(Frame, output,type);
            count = 0;

        }else if(type==4){
            cartoon(Frame, output);
            count = 0;
        }else if(type==5){
            sketch(Frame, output);
            count = 0;
        }else if(type==6){
            count++;
            output = Frame;
            if(count==1){
                faceSwap(Frame);
            }
        }
        // 使用图像处理技术将获取的帧经过处理后输入到output中
        namedWindow("Cartoon", CV_WINDOW_NORMAL);
        imshow("Cartoon", output);
        
    }
    
    
    return 0;
}

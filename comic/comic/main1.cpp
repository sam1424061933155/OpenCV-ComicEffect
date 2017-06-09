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


using namespace std;
using namespace cv;


int main(int argc, const char * argv[]) {
    
    int num_down=2;
    int num_bilateral = 7;

    
    Mat Image = imread("/Users/sam/Desktop/course/cv/final_project/pic_test.jpg",CV_LOAD_IMAGE_COLOR);
    Mat small_image(Image.rows,Image.cols,CV_8UC1);
    cout <<"Image size "<<Image.cols<<" , "<<Image.rows<<endl;
    //small_image =Image;
    Image.copyTo(small_image);

/*    for(int i=0;i<num_down;i++){
        pyrDown(small_image,small_image);

    }*/
    cout <<"samll image size "<<small_image.cols<<" , "<<small_image.rows<<endl;

    for(int i=0;i<num_bilateral;i++){
        Mat dst(small_image.rows,small_image.cols,CV_8UC1);
        cout << "dst dize"<<dst.cols<<" , "<<dst.rows<<endl;
        bilateralFilter(small_image,dst,9,9,7);
        small_image = dst;
    }
    cout <<"samll image size "<<small_image.cols<<" , "<<small_image.rows<<endl;

/*    for(int i=0;i<num_down;i++){
        pyrUp(small_image, small_image);
        
    }
    cout <<"samll image size "<<small_image.cols<<" , "<<small_image.rows<<endl;*/
    //Mat input(Image.rows,Image.cols,CV_8UC1);
    //small_image.copyTo(input)

    Mat img_gray;
    cvtColor(Image, img_gray, COLOR_BGRA2GRAY);
    Mat img_blur;
    Mat img_edge;

    medianBlur(img_gray, img_blur, 7);
    adaptiveThreshold(img_blur, img_edge, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 2);
    cvtColor(img_edge, img_edge, COLOR_GRAY2BGR);
    
    Mat img_cartoon;

    cout <<"samll image size "<<small_image.cols<<" , "<<small_image.rows<<endl;
    cout <<"img_edge size "<<img_edge.cols<<" , "<<img_edge.rows<<endl;

    bitwise_and(small_image, img_edge, img_cartoon);


    imshow("CV", img_cartoon);
    waitKey();
   
    
       return 0;
}

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
    
    Mat Image = imread("/Users/sam/Desktop/course/cv/final_project/pic_test.jpg", CV_LOAD_IMAGE_COLOR);
    Mat small_image(Image.cols,Image.rows,CV_32FC1);
    small_image =Image;
    
    //resize(Image,small_image,Size(Image.cols/2,Image.rows/2));
    cvtColor(small_image, small_image, COLOR_BGRA2GRAY);
    blur(small_image,small_image,Size(3,3));
    Canny(small_image,small_image,50, 150, 3);
    
    Mat kernel(3,3,CV_32FC1);
    for(int i=0;i<kernel.rows;i++){
        for(int j=0;j<kernel.cols;j++){
            kernel.at<float>(i,j)=1.0;
        }
    }
    kernel = kernel/12.0;
    filter2D(small_image, small_image, 0, kernel);
    
    threshold(small_image, small_image, 50, 255, THRESH_BINARY);  //not sure
    
    
    cvtColor(small_image, small_image, COLOR_GRAY2BGR);
    imshow("CV", small_image);
    waitKey();
   
    
    Mat shifted(Image.cols,Image.rows,CV_32FC1);
    pyrMeanShiftFiltering(small_image, shifted, 5, 20);
    //imshow("CV", shifted);
    //waitKey();
    //Mat output(3,3,CV_32FC1);
    Mat output;
    //small_image = shifted -small_image;
    subtract(shifted, small_image, output);
    
   
    
    
    
    imshow("CV", output);
    waitKey();
    
    return 0;
}

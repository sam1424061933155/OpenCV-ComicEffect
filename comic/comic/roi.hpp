//
//  roi.hpp
//  comicingimage
//
//  Created by 張修齊 on 2017/6/13.
//  Copyright © 2017年 張修齊. All rights reserved.
//


#ifndef ROI
#define ROI


#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>

using namespace cv;

class My_ROI{
public:
    My_ROI();
    My_ROI(Point upper_corner, Point lower_corner,Mat src);
    Point upper_corner, lower_corner;
    Mat roi_ptr;
    Scalar color;
    int border_thickness;
    void draw_rectangle(Mat src);
};



#endif


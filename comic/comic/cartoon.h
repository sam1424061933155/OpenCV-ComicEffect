//
//  cartoon.h
//  comic
//
//  Created by sam on 2017/6/8.
//  Copyright © 2017年 sam. All rights reserved.
//

#ifndef cartoon_h
#define cartoon_h
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class cartoon{
public: void cartoonTransform(cv::Mat &Frame, cv::Mat &output);
};


#endif /* cartoon_h */

#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img,double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, forw_img;
    vector<cv::Point2f> n_pts; // 新特征点
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts; // 原始特征点
    vector<cv::Point2f> prev_un_pts, cur_un_pts; // 去畸变后的特征点
    vector<cv::Point2f> pts_velocity; // 特征点的速度
    vector<int> ids; // 特征点的id
    vector<int> track_cnt; // 特征点的跟踪次数
    map<int, cv::Point2f> cur_un_pts_map; // 当前帧去畸变后的特征点映射（id 到特征点的映射）
    map<int, cv::Point2f> prev_un_pts_map; // 上一帧去畸变后的特征点映射（id 到特征点的映射）
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    static int n_id;
};

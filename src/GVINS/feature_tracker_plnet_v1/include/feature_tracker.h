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

#include "feature_detector.h"
#include "point_matcher.h"
#include "line_processor.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);
void reduceMatrix(Eigen::Matrix<float, 259, Eigen::Dynamic> &m, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    void setconfigs(VisualOdometryConfigs &configs);

    void readImage(const cv::Mat &_img,double _cur_time);

    void setMask();

    void addPoints();

    bool updateID_points(unsigned int i);

    bool updateID_lines(unsigned int i);

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
    vector<int> point_ids; // 特征点的id
    vector<int> point_track_cnt; // 特征点的跟踪次数
    map<int, cv::Point2f> cur_un_pts_map; // 当前帧去畸变后的特征点映射（id 到特征点的映射）
    map<int, cv::Point2f> prev_un_pts_map; // 上一帧去畸变后的特征点映射（id 到特征点的映射）
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    static int n_id_point;
    static int n_id_line;

    FeatureDetectorPtr feature_detector; // 特征检测器
    
    PointMatcherPtr point_matcher; // 点特征匹配器
    Eigen::Matrix<float, 259, Eigen::Dynamic> cur_features, forw_features; // 点特征描述子
    map<int, int> cur_features2pts, forw_features2pts;
    vector<int> cur_pts2features, forw_pts2features;
    std::vector<cv::DMatch> point_matches; // 点特征匹配结果


    std::vector<Eigen::Vector4d> cur_lines, forw_lines; // 线特征
    std::vector<std::map<int, double>> cur_pl_relation, forw_pl_relation; // 点特征和线特征的关系
    std::vector<int> line_matches; // 线特征匹配结果
    vector<int> line_ids; // 线特征的id
    vector<int> line_track_cnt; // 线特征的跟踪次数
};

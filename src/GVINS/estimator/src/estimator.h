#pragma once

#include <typeinfo>
#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include "initial/gnss_vi_initializer.h"
#include <std_msgs/Header.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"
#include "factor/gnss_psr_dopp_factor.hpp"
#include "factor/gnss_dt_ddt_factor.hpp"
#include "factor/gnss_dt_anchor_factor.hpp"
#include "factor/gnss_ddt_smooth_factor.hpp"
#include "factor/pos_vel_factor.hpp"
#include "factor/pose_anchor_factor.h"

#include <opencv2/core/eigen.hpp>

#include <gnss_comm/gnss_utility.hpp>
#include <gnss_comm/gnss_ros.hpp>
#include <gnss_comm/gnss_spp.hpp>

using namespace gnss_comm;

class Estimator
{
  public:
    Estimator();

    void setParameter();

    // interface

    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processGNSS(const std::vector<ObsPtr> &gnss_mea);
    void inputEphem(EphemBasePtr ephem_ptr);
    void inputIonoParams(double ts, const std::vector<double> &iono_params);
    void inputGNSSTimeDiff(const double t_diff);

    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header);

    // internal

    void clearState();
    bool initialStructure();
    bool visualInitialAlign();

    // GNSS related

    bool GNSSVIAlign();

    void updateGNSSStatistics();

    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    SolverFlag solver_flag; // 求解器标志
    MarginalizationFlag  marginalization_flag; // 边缘化标志
    Vector3d g; // 重力加速度
    // MatrixXd Ap[2];
    // VectorXd bp[2];

    Matrix3d ric[NUM_OF_CAM];  // 相机到 IMU 的旋转矩阵
    Vector3d tic[NUM_OF_CAM];  // 相机到 IMU 的平移向量

    Vector3d Ps[(WINDOW_SIZE + 1)];   // 滑动窗口中每一帧的位置
    Vector3d Vs[(WINDOW_SIZE + 1)];   // 滑动窗口中每一帧的速度
    Matrix3d Rs[(WINDOW_SIZE + 1)];   // 滑动窗口中每一帧的旋转矩阵
    Vector3d Bas[(WINDOW_SIZE + 1)];  // 滑动窗口中每一帧的加速度偏置
    Vector3d Bgs[(WINDOW_SIZE + 1)];  // 滑动窗口中每一帧的角速度偏置
    double td;  // 特征点的视差

    Matrix3d back_R0, last_R, last_R0; // back_R0: 被移除的最早图像帧的旋转矩阵; last_R: 滑动窗口中最后一帧图像的旋转矩阵; last_R0: 被移除的最新图像帧的旋转矩阵
    Vector3d back_P0, last_P, last_P0; // back_P0: 被移除的最早图像帧的位置; last_P: 滑动窗口中最后一帧图像的位置; last_P0: 被移除的最新图像帧的位置
    std_msgs::Header Headers[(WINDOW_SIZE + 1)]; // 滑动窗口中每一帧图像的头部信息

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)]; // 滑动窗口中每一帧的预积分对象
    Vector3d acc_0, gyr_0; // 当前帧的加速度和角速度

    vector<double> dt_buf[(WINDOW_SIZE + 1)]; // 滑动窗口中每一帧的时间间隔
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)]; // 滑动窗口中每一帧的加速度
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)]; // 滑动窗口中每一帧的角速度

    // GNSS related
    bool gnss_ready; // GNSS 数据是否准备好
    Eigen::Vector3d anc_ecef; // 基站的 ECEF 坐标
    Eigen::Matrix3d R_ecef_enu; // ECEF 到 ENU 的旋转矩阵
    double yaw_enu_local; // 本地 ENU 坐标系的偏航角
    std::vector<ObsPtr> gnss_meas_buf[(WINDOW_SIZE+1)]; // 滑动窗口中每一帧的 GNSS 观测数据
    std::vector<EphemBasePtr> gnss_ephem_buf[(WINDOW_SIZE+1)]; // 滑动窗口中每一帧的 GNSS 星历数据
    std::vector<double> latest_gnss_iono_params; // 最新的 GNSS 电离层参数
    std::map<uint32_t, std::vector<EphemBasePtr>> sat2ephem; // 卫星 ID 到星历数据的映射
    std::map<uint32_t, std::map<double, size_t>> sat2time_index; // 卫星 ID 到时间索引的映射
    std::map<uint32_t, uint32_t> sat_track_status; // 卫星 ID 到跟踪状态的映射
    double para_anc_ecef[3]; // 基站的 ECEF 坐标
    double para_yaw_enu_local[1]; // 本地 ENU 坐标系的偏航角
    double para_rcv_dt[(WINDOW_SIZE+1)*4]; // 滑动窗口中每一帧接收机的时间偏差
    double para_rcv_ddt[WINDOW_SIZE+1]; // 滑动窗口中每一帧接收机的时间偏差变化率
    // GNSS statistics
    double diff_t_gnss_local; // GNSS 本地时间差
    Eigen::Matrix3d R_enu_local; // ENU 到本地 ENU 的旋转矩阵
    Eigen::Vector3d ecef_pos, enu_pos, enu_vel, enu_ypr; // ECEF 坐标、ENU 坐标、ENU 速度、ENU 姿态

    int frame_count; // 当前帧的 ID
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid; // 外点数、后端移动次数、前端移动次数、无效次数

    FeatureManager f_manager; // 特征管理器
    MotionEstimator m_estimator; // 运动估计器
    InitialEXRotation initial_ex_rotation; // 初始外部旋转估计器

    bool first_imu; // 是否是第一帧 IMU 数据
    bool is_valid, is_key; // 是否有效、是否是关键帧
    bool failure_occur; // 是否发生故障

    vector<Vector3d> point_cloud; // 点云
    vector<Vector3d> margin_cloud; // 边缘化点云
    vector<Vector3d> key_poses; // 关键帧的位置
    double initial_timestamp; // 第一帧图像的时间戳

    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE]; // 滑动窗口中每一帧的位姿
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS]; // 滑动窗口中每一帧的速度和偏置
    double para_Feature[NUM_OF_F][SIZE_FEATURE]; // 特征点
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];  // 相机的外部位姿
    double para_Td[1][1];   // 特征点的视差

    MarginalizationInfo *last_marginalization_info; // 上一次边缘化信息
    vector<double *> last_marginalization_parameter_blocks; // 上一次边缘化参数块

    map<double, ImageFrame> all_image_frame; // 保存滑动窗口内所有图像帧的特征点信息
    IntegrationBase *tmp_pre_integration; // 临时预积分对象

    bool first_optimization; // 是否是第一次优化
};

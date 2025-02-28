#include "feature_tracker.h"

int FeatureTracker::n_id_point = 0;
int FeatureTracker::n_id_line = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

// 删除status为0的元素
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceMatrix(Eigen::Matrix<float, 259, Eigen::Dynamic> &m, vector<uchar> status)
{
    Eigen::Matrix<float, 259, Eigen::Dynamic> temp;
    int j = 0;
    for (int i = 0; i < int(m.cols()); i++)
        if (status[i])
        {
            temp.resize(259, j + 1);
            temp.col(j++) = m.col(i);
        }     
    m = temp;
}



FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setconfigs(VisualOdometryConfigs &configs)
{
    feature_detector = std::shared_ptr<FeatureDetector>(new FeatureDetector(configs.plnet_config));
    point_matcher = std::shared_ptr<PointMatcher>(new PointMatcher(configs.point_matcher_config));
}

/**
 * @brief 根据特征点的跟踪历史和位置信息，更新掩膜，并筛选出需要保留的特征点
 */
void FeatureTracker::setMask()
{
    // 如果是鱼眼相机，使用鱼眼相机的掩码，否则使用全白的掩码
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    
    // 优先保留跟踪时间长的特征点
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(point_track_cnt[i], make_pair(forw_pts[i], point_ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    point_ids.clear();
    point_track_cnt.clear();

    // 更新掩膜
    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255) // 检查当前特征点在掩膜中的值是否为 255（表示该位置允许检测特征点）
        {
            forw_pts.push_back(it.second.first); // 保存特征点
            point_ids.push_back(it.second.second); // 保存特征点的id
            point_track_cnt.push_back(it.first); // 保存特征点的跟踪次数
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1); // 在掩膜中将特征点的周围区域置为 0, 表示该区域不允许检测特征点
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        point_ids.push_back(-1);
        point_track_cnt.push_back(1);
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    if (EQUALIZE)
    {
        // 对比度限制自适应直方图均衡化，增强图像的对比度
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8)); // 对比度限制阈值为3.0，设置每个区域的大小为8*8
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();
    forw_features2pts.clear();

    // 采用plnet检测下一帧点线特征
    TicToc t_t;
    Eigen::Matrix<float, 259, Eigen::Dynamic> features;
    std::vector<Eigen::Vector4d> lines;
    feature_detector->Detect(forw_img, features, lines);
    forw_features = features;
    forw_lines = lines;
    ROS_DEBUG("detect feature costs: %fms", t_t.toc());

    if(cur_features.cols() > 0)
    {
        /****** 点特征跟踪 ******/

        // 采用super glue跟踪特征点
        TicToc t_o;
        point_matches.clear();
        if(!point_matcher->MatchingPoints(cur_features, forw_features, point_matches, true))
            ROS_WARN("MatchingPoints failed");
        ROS_DEBUG("super glue costs: %fms", t_o.toc());
        
        //记录跟踪状态status，将成功跟踪的特征点保存到forw_pts中，并将features中该特征点得分置-1
        vector<uchar> status(cur_pts.size(), 0);
        forw_pts = cur_pts;
        for (unsigned int i = 0; i < point_matches.size(); i++)
        {
            if (point_matches[i].distance < 0.7) // todo: 距离阈值，可调
            {
                auto idx = cur_features2pts.find(point_matches[i].queryIdx);
                if(idx == cur_features2pts.end())
                    continue;
                int cur_pts_idx = idx->second;
                status[cur_pts_idx] = 1;
                cv::Point2f pt(features(1, point_matches[i].trainIdx), features(2, point_matches[i].trainIdx));
                forw_pts[cur_pts_idx] = pt;
                forw_pts2features[cur_pts_idx] = point_matches[i].trainIdx;
                features(0, point_matches[i].trainIdx) = -1;
            }
        }
        
        // 根据status，更新prev_pts、cur_pts、ids、cur_un_pts、track_cnt、cur_features、forw_features
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(point_ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(point_track_cnt, status);
        reduceVector(cur_pts2features, status);
        reduceVector(forw_pts2features, status);

        // 更新forw_features2pts
        for(int i = 0; i < forw_pts2features.size(); i++)
        {
            forw_features2pts.insert(make_pair(forw_pts2features[i], i));
        }

        /****** 线特征跟踪 ******/

        line_matches.clear();
        forw_pl_relation.clear();
        if(forw_lines.size() > 0 && forw_features.cols() > 0)
        {
            // 关联当前的线特征和点特征
            AssignPointsToLines(forw_lines, forw_features, forw_pl_relation);
            if(cur_lines.size() > 0)
            {
                // 匹配线特征
                MatchLines(cur_pl_relation, forw_pl_relation, point_matches, cur_features.cols(), forw_features.cols(), line_matches);
            }
        }
    }

    // 更新点特征跟踪次数
    for (auto &n : point_track_cnt)
        n++;

    // 更新线特征跟踪次数和id
    vector<int> n_line_track_cnt(forw_lines.size(), 1);
    vector<int> n_line_ids(forw_lines.size(), -1);
    for (int i = 0; i < line_matches.size(); i++)
    {
        if(line_matches[i] != -1)
        {
            int j = line_matches[i];
            n_line_track_cnt[j] += line_track_cnt[i];
            n_line_ids[j] = line_ids[i];
        }
    }
    line_track_cnt = n_line_track_cnt;
    line_ids = n_line_ids;

    // 如果现有特征点数量小于最大特征点数量，添加新的特征点
    if (PUB_THIS_FRAME)
    {
        ROS_DEBUG("add feature begins");
        TicToc t_a;

        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            // 根据匹配度排序，取前 MAX_CNT - forw_pts.size() 个特征点
            std::vector<std::pair<float, std::pair<int, cv::Point2f>>> features_sorted;
            for (int i = 0; i < features.cols(); i++)
            {
                float score = features(0, i);
                if(score < 0)
                    continue;
                cv::Point2f pt(features(1, i), features(2, i));
                int index = i;
                features_sorted.push_back(std::make_pair(score, std::make_pair(index, pt)));
            }
            std::sort(features_sorted.begin(), features_sorted.end(), [](const auto &a, const auto &b) {
                return a.first > b.first;
            });

            n_pts.clear();
            int features_cnt = min(MAX_CNT - forw_pts.size(), features_sorted.size());
            for(int i = 0; i < features_cnt; i++)
            {
                cv::Point2f pt = features_sorted[i].second.second;
                n_pts.push_back(pt);
                forw_pts2features.push_back(features_sorted[i].second.first);
                forw_features2pts.insert(make_pair(features_sorted[i].second.first, forw_pts2features.size() - 1));
            }
        }
        else{
            n_pts.clear();
        }

        // 添加新的特征点
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }

    // 迭代，并计算特征点的速度
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
    cur_features = forw_features;
    cur_pts2features = forw_pts2features;
    cur_features2pts = forw_features2pts;
    cur_lines = forw_lines;
    cur_pl_relation = forw_pl_relation;
}

/**
 * @brief 使用 RANSAC 算法计算基础矩阵，并根据基础矩阵剔除异常点
 */
void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(point_ids, status);
        reduceVector(point_track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID_points(unsigned int i)
{
    if (i < point_ids.size())
    {
        if (point_ids[i] == -1)
            point_ids[i] = n_id_point++;
        return true;
    }
    else
        return false;
}

bool FeatureTracker::updateID_lines(unsigned int i)
{
    if (i < line_ids.size())
    {
        if (line_ids[i] == -1)
            line_ids[i] = n_id_line++;
        return true;
    }
    else
        return false;
}


void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

/**
 * @brief 去畸变，并计算特征点的速度
 */
void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b); // 将像素坐标转化为无畸变的归一化坐标
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(point_ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", point_ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }

    // 计算特征点的速度，速度的计算方式为当前帧特征点的位置减去上一帧特征点的位置除以时间间隔，如果特征点在上一帧中没有出现，则速度为0
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (point_ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(point_ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}

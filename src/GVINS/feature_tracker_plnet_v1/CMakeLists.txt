cmake_minimum_required(VERSION 2.8.3)
project(gvins_feature_tracker_plnet_v1)

#set(CMAKE_BUILD_TYPE "Release")
#(CMAKE_CXX_FLAGS "-std=c++11")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "release")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
add_definitions(-w)

# add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/tensorrtbuffer)
# add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/DBoW2)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    gvins_camera_model

    geometry_msgs
    image_transport
    nav_msgs
    rospy
    tf
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED)
find_package(G2O REQUIRED)
find_package(Gflags REQUIRED)
find_package(Glog REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES ${PROJECT_NAME}_lib
 CATKIN_DEPENDS geometry_msgs image_transport nav_msgs roscpp rospy std_msgs
)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include

    ${PROJECT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIR}
    ${GFLAGS_INCLUDE_DIRS} 
    ${GLOG_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_lib SHARED
    src/feature_detector.cc
    src/plnet.cpp
    src/super_point.cpp
    src/utils.cc
    src/point_matcher.cc
    src/super_glue.cpp
    src/light_glue.cpp
    src/line_processor.cc
    src/camera.cc
)

target_link_libraries(${PROJECT_NAME}_lib
  nvinfer
  nvonnxparser
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${Boost_LIBRARIES}
  ${G2O_LIBRARIES}
  ${GFLAGS_LIBRARIES} 
  ${GLOG_LIBRARIES}
  yaml-cpp
  tensorrtbuffer
  DBoW2
  -lboost_serialization
)

add_executable(gvins_feature_tracker_plnet_v1
    src/feature_tracker_node.cpp
    src/feature_tracker.cpp
    src/parameters.cpp
)

target_link_libraries(gvins_feature_tracker_plnet_v1 ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_NAME}_lib)
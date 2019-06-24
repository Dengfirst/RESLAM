/**
* This file is part of RESLAM.
*
* Copyright (C) 2014-2019 Schenk Fabian <schenk at icg dot tugraz dot at> (Graz University of Technology)
* For more information see <https://github.com/fabianschenk/RESLAM/>
*
* RESLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* RESLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with RESLAM. If not, see <http://www.gnu.org/licenses/>.
* 
*
*  If you use this software please cite at least one of the following publications:
*  - RESLAM: A robust edge-based SLAM system,  Schenk Fabian, Fraundorfer Friedrich, ICRA 2019
*  - Robust Edge-based Visual Odometry using Machine-Learned Edges, Schenk Fabian, Fraundorfer Friedrich, IROS 2017
*  - Combining Edge Images and Depth Maps for Robust Visual Odometry, Schenk Fabian, Fraundorfer Friedrich, BMVC 2017
*/
#include <thread>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Input.h"
#include "../Utils/timer.h"
#include "../Utils/Logging.h"
#include "../config/Defines.h"
#include "../System/SystemSettings.h"

#ifdef WITH_ORBBEC_ASTRA_PRO
    #include "./orbbec_astra/InputOrbbecAstra.h"
#endif
namespace RESLAM
{

PoseVector
Input::readGTTrajectory()
{
    PoseVector gtPoses;
    std::ifstream fileGtPoses;
    const std::string gtFileName = (mSystemSettings.InputDatasetFolder+"/groundtruth.txt");
    fileGtPoses.open(gtFileName.c_str(),std::ios_base::in);
    I3D_LOG(i3d::info) << "Reading: " << gtFileName;
    if (!fileGtPoses.is_open())
    {
        I3D_LOG(i3d::error) <<"Could not open file list: " << gtFileName;
    }

    std::string line;
    SE3Pose firstPose;
    while(std::getline(fileGtPoses,line))
    {
        //unsigned long long int utime;
        double utime;
        float x, y, z, qx, qy, qz, qw;

        if (line[0] == '#') //skip comments
            continue;
        int n = sscanf(line.c_str(), "%lf %f %f %f %f %f %f %f", &utime, &x, &y, &z, &qx, &qy, &qz, &qw);
        if (n!=8)
        {
            I3D_LOG(i3d::info) << "n = " << n;
            exit(0);
        }
        Eigen::Quaterniond q(qw, qx, qy, qz);
        Eigen::Vector3d t(x, y, z);
        if (gtPoses.empty())
        {
            firstPose = SE3Pose(q,t);
        }
        gtPoses.emplace_back(firstPose.inverse()*SE3Pose(q,t));
    }
    fileGtPoses.close();
    return gtPoses;
}

InputDatasetReader::InputDatasetReader(const SystemSettings& settings):
    Sensor(settings), mFileName(settings.InputDatasetFolder+"/"+settings.InputDatasetFile)
{
    //TODO: Manage different input types
    mInputFormat = [input = settings.InputFormat]()
                    { switch (input)
                        {
                        case 1: return InputFormat::BundleFusion; //currently not supported
                        case 2: return InputFormat::Files; //currently not supported
                        case 0:
                        default: return InputFormat::TUM;
                        }}();

    switch (mInputFormat)
    {
        case InputFormat::Files:
        case InputFormat::BundleFusion:
        case InputFormat::TUM:
        default:
            mFileList.open(mFileName.c_str(),std::ios_base::in);
            I3D_LOG(i3d::info) << "Reading: " << mFileName;
                if (!mFileList.is_open())
                {
                    I3D_LOG(i3d::error) <<"Could not open file list: " << mFileName << " xx: " << settings.InputDatasetFile;
                    exit(0);
                }
            assert(mFileList.is_open() && "File could not been opened!");
            mFlagIsRunning = true;
    }
}
InputDatasetReader::~InputDatasetReader()
{
    if (mFileList.is_open()) mFileList.close();
}

/**
 * [InputDatasetReader::getImages 从数据集读取图像序列]
 * @param rgb       [ ]
 * @param depth     [ ]
 * @param timestamp [ ]
**/
bool 
InputDatasetReader::getImages(cv::Mat& rgb,cv::Mat& depth, double& timestamp)
{
    std::string currRGBFile, currDepthFile;
    std::string inputLine;
    double rgbTimeStamp = 0, depthTimeStamp = 0;
    //read lines
    while(std::getline(mFileList,inputLine) && mFlagIsRunning)
    {
        I3D_LOG(i3d::info) << "getimages!" << mNbReceivedImages;
        //ignore comments
        if (inputLine[0] == '#' || inputLine.empty()) continue;
        ++mNbReceivedImages;
        I3D_LOG(i3d::info) << "mNbReceivedImages!" << mNbReceivedImages;

        // 设置数据集开始帧读取位置
        if (mNbReceivedImages < mSystemSettings.InputSkipFirstNFrames) continue;
        std::istringstream is_associate(inputLine);
        is_associate >> rgbTimeStamp >> currRGBFile >> depthTimeStamp >> currDepthFile; // 载入 associate 文件
        // 读取 RGB 和 depth 图片
        rgb = cv::imread(mSystemSettings.InputDatasetFolder+"/"+currRGBFile);
        depth = cv::imread(mSystemSettings.InputDatasetFolder+"/"+currDepthFile,CV_LOAD_IMAGE_UNCHANGED);
        //divide by 5000 to get distance in metres
        depth.convertTo(depth,CV_32FC1,1.0f/mSystemSettings.InputDepthScaleFactor);
        timestamp =  depthTimeStamp;
        cv::Mat gray;
        cv::cvtColor(rgb,gray,CV_BGRA2GRAY);
        return true;
    }
    // 数据集读取完了
    LOG_THRESHOLD(i3d::info);
    I3D_LOG(i3d::info) << "Stopping reading images!";
    LOG_THRESHOLD(i3d::info);
    mFlagIsRunning = false;
    return false;
}

/**
 * 1、传感器类型SensorType：以枚举的方式，包含 Dataset(默认)、OrbbecAstra、OrbbecAstraPro、RealsenseZR300、RealsenseD4XX
 * 2、图像读取
**/
void 
Input::startInput()
{
    const SensorType sensorType = intToSensorType(mSystemSettings.InputSensorType);
    mSensor = [sensorType,this]() -> std::unique_ptr<Sensor>{
                                switch (sensorType)
                                {
                                    case SensorType::Dataset: return std::make_unique<InputDatasetReader>(mSystemSettings);
                                    #ifdef WITH_ORBBEC_ASTRA_PRO
                                        case SensorType::OrbbecAstraPro: return std::make_unique<InputOrbbecAstraProReader>(mSystemSettings);
                                    #endif
                                    #ifdef WITH_ORBBEC_ASTRA
                                        case SensorType::OrbbecAstra: return std::make_unique<InputOrbbecAstraReader>(mSystemSettings);
                                    #endif
                                    default: return std::make_unique<InputDatasetReader>(mSystemSettings);
                                }
                            }();
    std::thread th(&Input::readImages,this);
    th.detach(); //! 分离式线程
}

void
Input::stopInput()
{
    I3D_LOG(i3d::info) << "Stopping Input Sensor";
    mSensor->stopSensor();
}

/**
 * 读取图像
 *
**/
void 
Input::readImages()
{
    cv::Mat gray,depth,rgb;
    if (intToSensorType(mSystemSettings.InputSensorType) != SensorType::Dataset)
    {
        rgb = cv::Mat(mSystemSettings.IntrinsicsImgSize,CV_8UC3);
        depth = cv::Mat(mSystemSettings.IntrinsicsImgSize,CV_32FC1);
    }
    while(true)
    {
        I3D_LOG(i3d::nothing) << "readImages loop!";
        double timestamp;
        //! mInputFrameQueue图像序列保持10帧，超过10帧线程解锁，进行读取操作
        {
            std::unique_lock<std::mutex> lock(mInputFrameQueueMutex);
            if (mInputFrameQueue.size() > 10)
            {
                I3D_LOG(i3d::info) << "size: " << mInputFrameQueue.size();
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                continue;
            }
        }
        //read 注意这里的时间戳用的是 depth 采集的时间
        if (!mSensor->getImages(rgb,depth,timestamp) || mNbOfImages > mSystemSettings.InputReadNFrames) break;
        I3D_LOG(i3d::nothing) << "Got image from sensor!" << mNbOfImages;
        //convert to gray
        cv::cvtColor(rgb,gray,(mSystemSettings.InputColorFormatRGB ? CV_RGB2GRAY : CV_BGR2GRAY));
            
        //add to queue
        //! 计算深度图Jet,Canny边缘检测, come to FrameSet::FrameSet
        auto nextFrame = std::make_unique<FrameSet>(gray,depth,rgb,timestamp,mSystemSettings);
        {
            std::lock_guard<std::mutex> lock(mInputFrameQueueMutex);
            mInputFrameQueue.push(std::move(nextFrame));
            I3D_LOG(i3d::nothing) << "mInputFrameQueue: " << mInputFrameQueue.size();
        }
        mNbOfImages++;
        I3D_LOG(i3d::nothing) << "mNbOfImages: " << mNbOfImages;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    I3D_LOG(i3d::nothing) << "Stopping readImages!";
    mSensor->stopSensor();
    I3D_LOG(i3d::nothing) << "Stopping readImages! stopSensor()";
}
}



#include <math.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
// #include <std_srvs/Trigger.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>

#include <dynamic_reconfigure/server.h>
#include <navtech_projection/paramsConfig.h>

using namespace std;
using namespace cv;

class RadarImageConverter
{
  ros::NodeHandle nh;
  image_transport::ImageTransport it;
  image_transport::Subscriber sub_image;
  ros::Publisher pub_pc;
  ros::Publisher pub_image;

  pcl::PCLPointCloud2::Ptr pc;
  sensor_msgs::PointCloud2 ros_pc2;
  Rect cart_roi;
  float radar_distance_resolution, cart_max_range, pc_max_range, pc_pixel_resolution;
  int cart_image_size, pc_intensity_thres;
  bool mapUpdated, pc_enable, cart_enable;
  Mat mapX, mapY;

  dynamic_reconfigure::Server<navtech_projection::paramsConfig> server;
  dynamic_reconfigure::Server<navtech_projection::paramsConfig>::CallbackType f;

public:
  RadarImageConverter()
      : nh("~"), it(nh), cart_image_size(5712),
        mapUpdated(false)
  {

    nh.param<float>("radar_distance_resolution", radar_distance_resolution, 0.16);

    sub_image = it.subscribe("/Navtech/Polar", 1, &RadarImageConverter::imageCb, this);
    pub_pc = nh.advertise<sensor_msgs::PointCloud2>("/Navtech/Points", 1);
    pub_image = nh.advertise<sensor_msgs::Image>("/Navtech/Cartesian", 1);

    mapX = cv::Mat::zeros(cart_image_size, cart_image_size, CV_32FC1);
    mapY = cv::Mat::zeros(cart_image_size, cart_image_size, CV_32FC1);

    f = boost::bind(&RadarImageConverter::dynamic_reconf_cb, this, _1, _2);
    server.setCallback(f);
  }

  void dynamic_reconf_cb(navtech_projection::paramsConfig &config, uint32_t level)
  {
    ROS_INFO("Updating params");
    cart_enable = config.enable_cartesian;
    cart_image_size = config.cart_image_size;
    cart_max_range = config.cart_max_range;

    pc_enable = config.enable_pointcloud;
    pc_max_range = config.pc_max_range;
    pc_pixel_resolution = config.pc_pixel_resolution;
    pc_intensity_thres = config.pc_intensity_threshold;

    cart_roi.x = mapX.cols/2 - cart_max_range/radar_distance_resolution;
    cart_roi.y = mapX.rows/2 - cart_max_range/radar_distance_resolution;
    cart_roi.width = cart_max_range/radar_distance_resolution*2;
    cart_roi.height = cart_max_range/radar_distance_resolution*2;
    
  }

  void updateMaps(int in_rows, int in_cols, int out_size)
  {
    ROS_INFO("Updateing remap values...");
    mapX = cv::Mat::zeros(out_size, out_size, CV_32FC1);
    mapY = cv::Mat::zeros(out_size, out_size, CV_32FC1);

    int x, y;
    for (int i = 0; i < out_size; i++)
    {
      float *ptrX = mapX.ptr<float>(i);
      float *ptrY = mapY.ptr<float>(i);
      for (int j = 0; j < out_size; j++)
      {
        x = i - out_size / 2;
        y = j - out_size / 2;
        ptrY[j] = sqrt(pow((float)x, 2) + pow((float)y, 2)) * 2 / out_size * in_rows;

        if (x < 0 && y >= 0)
          ptrX[j] = atan((float)-y / x);
        else if (x >= 0 && y >= 0)
          ptrX[j] = (atan((float)x / y) + M_PI / 2);
        else if (x >= 0 && y < 0)
          ptrX[j] = (atan((float)-y / x) + M_PI);
        else
          ptrX[j] = (atan((float)x / y) + M_PI / 2 * 3);
        ptrX[j] *= in_cols / (2 * M_PI);
        ptrX[j] = round(ptrX[j]) + 1;
      }
    }
    ROS_INFO("Remap values initialized...");
  }

  void imagePolartoCart(cv_bridge::CvImagePtr img_in, cv_bridge::CvImagePtr img_out)
  {
    if (!mapUpdated)
    {
      updateMaps(img_in->image.rows, img_in->image.cols, img_in->image.rows * 2);
      mapUpdated = true;
    }
    Mat crossover(img_in->image.rows, img_in->image.cols + 2, CV_8UC1);
    cv::hconcat(img_in->image.col(img_in->image.cols - 1), img_in->image, crossover);
    cv::hconcat(crossover.colRange(0, img_in->image.cols + 1), img_in->image.col(0), crossover);
    cv::remap(crossover, img_out->image, mapX, mapY, cv::INTER_LINEAR);

    return;
  }

  void imageCartToPC(cv_bridge::CvImagePtr cv_ptr,
                     pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out)
  {
    uint16_t half_side_length = pc_max_range / pc_pixel_resolution;
    uint16_t new_img_size = cv_ptr->image.rows * radar_distance_resolution / pc_pixel_resolution;
    uint16_t img_mid = new_img_size / 2;
    Mat resized_img(new_img_size, new_img_size, CV_8UC1);

    cv::resize(cv_ptr->image, resized_img, cv::Size(new_img_size, new_img_size));

    // ROS_INFO("generating point cloud");
    pcl::PointXYZI point;
    for (int i = -half_side_length; i < half_side_length; i++)
    {
      for (int j = -half_side_length; j < half_side_length; j++)
      {
        uint8_t point_value = resized_img.at<uint8_t>(img_mid + i, img_mid + j);
        if (point_value <= pc_intensity_thres)
          continue;

        point.x = -(float)i * pc_pixel_resolution;
        point.y = -(float)j * pc_pixel_resolution;
        point.z = 0;
        point.intensity = point_value;
        pc_out->points.push_back(point);
      }
    }
    ROS_INFO("pc->size():%zu", pc_out->size());
    return;
  }

  void imageCb(const sensor_msgs::ImageConstPtr &msg)
  {
    cv_bridge::CvImagePtr cv_ptr, cv_remapped(new cv_bridge::CvImage);

    // ROS_INFO("got polar image");
    try
    {
      // Convert to CV 8UC1 which is a gray-scale image
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC1);
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    imagePolartoCart(cv_ptr, cv_remapped);
    if (cart_enable)
    {

      sensor_msgs::ImagePtr img_msg_out;
      cv_bridge::CvImagePtr cv_resized(new cv_bridge::CvImage);

      cv::resize(cv_remapped->image(cart_roi), cv_resized->image, cv::Size(cart_image_size, cart_image_size));
      img_msg_out = cv_resized->toImageMsg();
      img_msg_out->encoding = sensor_msgs::image_encodings::MONO8;
      img_msg_out->header = msg->header;
      img_msg_out->header.frame_id = "navtech_optical";
      pub_image.publish(img_msg_out);
    }

    if (pc_enable)
    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_ptr(new pcl::PointCloud<pcl::PointXYZI>);
      sensor_msgs::PointCloud2 pc_msg_out;
      imageCartToPC(cv_remapped, pc_ptr);
      pcl::toROSMsg(*pc_ptr, pc_msg_out);
      pc_msg_out.is_dense = false;
      pc_msg_out.header = msg->header;
      pc_msg_out.header.frame_id = "navtech";
      pub_pc.publish(pc_msg_out);
    }
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "radar_image_projection");
  RadarImageConverter ic;
  ros::spin();
  return 0;
}

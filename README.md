# navtech_projection

## Supported Models
- Navtech CTS350-X
- Navtech CIR504-X
- Navtech CIR304-H

---
## Dependencies
- [ROS](http://wiki.ros.org/ROS/Tutorials)
- [PCL](https://pointclouds.org/)
- OpenCV (Should be installed during ROS installation)

## ROS Package Dependencies
- cv_bridge
- pcl_conversions
- pcl_ros
- sensor_msgs
- dynamic_reconfigure
- tf2_ros

## Subscribed topic
/Navtech/Polar (sensor_msgs/Image)

## Published topic
- /Navtech/Remapped (sensor_msgs/Image)
- /Navtech/Cartesian (sensor_msgs/Image) ( if cart_enable )
- /Navtech/Points (sensor_msgs/PointCloud2) ( if pc_enable )

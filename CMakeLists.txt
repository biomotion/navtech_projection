cmake_minimum_required(VERSION 3.0.2)
project(navtech_projection)


find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf2_ros
  dynamic_reconfigure
)
find_package(OpenCV REQUIRED)


## Generate dynamic reconfigure parameters in the 'cfg' folder
generate_dynamic_reconfigure_options(
  cfg/projection_params.cfg
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES navtech_projection
#  CATKIN_DEPENDS OpenCV cv_bridge pcl_conversions pcl_ros roscpp rospy sensor_msgs std_msgs tf2_ros
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(polar_to_cart src/polar_to_cart.cpp)
target_link_libraries(polar_to_cart
  ${catkin_LIBRARIES}
)

add_dependencies(polar_to_cart ${PROJECT_NAME}_gencfg)
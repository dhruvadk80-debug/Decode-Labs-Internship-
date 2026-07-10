cmake_minimum_required(VERSION 3.0.2)
project(ur5_ik_planner)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  geometry_msgs
  sensor_msgs
  trajectory_msgs
  visualization_msgs
  tf2_ros
  tf2_geometry_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
    roscpp
    std_msgs
    geometry_msgs
    sensor_msgs
    trajectory_msgs
    visualization_msgs
    tf2_ros
    tf2_geometry_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/ik_planner_node.py
  scripts/send_goal.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(DIRECTORY rviz/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
)

install(DIRECTORY worlds/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/worlds
)

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  catkin_add_nosetests(test/test_kinematics.py)
endif()

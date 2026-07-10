#!/usr/bin/env python3
"""
send_goal.py
============
Small command-line helper to publish a single goal pose to
/ur5_ik_planner/goal_pose, for manually testing the planner without
writing a custom client.

Usage
-----
    rosrun ur5_ik_planner send_goal.py X Y Z [QX QY QZ QW]

If the quaternion is omitted, identity orientation (no rotation) is used.

Example
-------
    rosrun ur5_ik_planner send_goal.py 0.4 0.1 0.3
    rosrun ur5_ik_planner send_goal.py 0.4 0.1 0.3 0 0 0 1
"""

import sys
import rospy
from geometry_msgs.msg import PoseStamped


def main():
    args = sys.argv[1:]
    if len(args) not in (3, 7):
        print(__doc__)
        sys.exit(1)

    x, y, z = (float(v) for v in args[:3])
    if len(args) == 7:
        qx, qy, qz, qw = (float(v) for v in args[3:7])
    else:
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

    rospy.init_node("send_goal", anonymous=True)
    pub = rospy.Publisher("/ur5_ik_planner/goal_pose", PoseStamped, queue_size=1, latch=True)
    rospy.sleep(0.5)  # give the publisher time to connect

    msg = PoseStamped()
    msg.header.frame_id = "base_link"
    msg.header.stamp = rospy.Time.now()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw

    pub.publish(msg)
    rospy.loginfo(f"Published goal pose: pos=({x},{y},{z}) quat=({qx},{qy},{qz},{qw})")
    rospy.sleep(0.5)


if __name__ == "__main__":
    main()

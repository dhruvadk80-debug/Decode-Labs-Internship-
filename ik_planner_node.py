#!/usr/bin/env python3
"""
ik_planner_node.py
===================
ROS node that ties together kinematics.py, collision.py, and
trajectory.py to move a simulated UR5 (in Gazebo) from its current
joint state to a requested Cartesian goal pose, avoiding obstacles
defined in the planning scene.

Subscribes
----------
/ur5_ik_planner/goal_pose (geometry_msgs/PoseStamped)
    Desired tool pose (Point B) in the 'base_link' frame. Receiving a
    message on this topic triggers a new plan + execute cycle.

/joint_states (sensor_msgs/JointState)
    Used to track the robot's current joint configuration (Point A),
    kept up to date continuously.

Publishes
---------
/eff_joint_traj_controller/command (trajectory_msgs/JointTrajectory)
    The planned, collision-free joint trajectory, sent to ROS-Industrial's
    standard joint trajectory controller (adjust the topic name to match
    your controller_manager config if different -- check with
    `rostopic list | grep trajectory`).

/ur5_ik_planner/obstacle_markers (visualization_msgs/MarkerArray)
    RViz markers for the obstacles currently registered in the planning
    scene, so you can see what the planner is avoiding.

Parameters (ros param server, set via launch file)
----------------------------------------------------
~trajectory_duration (float, default 4.0)   - seconds for planned moves
~n_samples (int, default 100)               - waypoints per trajectory
~max_bend_iterations (int, default 5)       - obstacle-avoidance retries
~obstacles (list of dicts, default [])      - static obstacles, each
    {x, y, z, radius, name}, loaded once at startup from the param
    server (see config/obstacles.yaml).
"""

import rospy
import numpy as np

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from ur5_ik_planner.kinematics import forward_kinematics
from ur5_ik_planner.collision import Environment
from ur5_ik_planner.trajectory import plan_path, TrajectoryPlanningError

# Standard UR5 joint names, in the order this package's kinematics.py
# expects them (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2,
# wrist_3). If your URDF / controller uses a different order, this is
# the place to remap it.
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def pose_msg_to_matrix(pose_stamped):
    """Convert a geometry_msgs/PoseStamped into a 4x4 homogeneous transform."""
    p = pose_stamped.pose.position
    q = pose_stamped.pose.orientation
    x, y, z, w = q.x, q.y, q.z, q.w

    # quaternion -> rotation matrix
    R = np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [p.x, p.y, p.z]
    return T


class UR5IKPlannerNode:
    def __init__(self):
        rospy.init_node("ur5_ik_planner_node")

        self.duration = rospy.get_param("~trajectory_duration", 4.0)
        self.n_samples = rospy.get_param("~n_samples", 100)
        self.max_bend_iterations = rospy.get_param("~max_bend_iterations", 5)
        traj_topic = rospy.get_param(
            "~trajectory_topic", "/eff_joint_traj_controller/command"
        )

        self.current_q = None  # populated from /joint_states
        self.joint_index_map = None

        self.env = self._load_obstacles_from_param_server()

        self.traj_pub = rospy.Publisher(traj_topic, JointTrajectory, queue_size=1)
        self.marker_pub = rospy.Publisher(
            "/ur5_ik_planner/obstacle_markers", MarkerArray, queue_size=1, latch=True
        )

        rospy.Subscriber("/joint_states", JointState, self._joint_state_cb)
        rospy.Subscriber("/ur5_ik_planner/goal_pose", PoseStamped, self._goal_pose_cb)

        self._publish_obstacle_markers()

        rospy.loginfo(
            "ur5_ik_planner_node ready. Waiting for /joint_states and "
            "goal poses on /ur5_ik_planner/goal_pose ..."
        )

    def _load_obstacles_from_param_server(self):
        env = Environment()
        obstacle_list = rospy.get_param("~obstacles", [])
        for obs in obstacle_list:
            env.add(
                center=[obs["x"], obs["y"], obs["z"]],
                radius=obs.get("radius", 0.05),
                name=obs.get("name", "obstacle"),
            )
        rospy.loginfo(f"Loaded {len(env.obstacles)} obstacle(s) into planning scene.")
        return env

    def _publish_obstacle_markers(self):
        arr = MarkerArray()
        for i, obs in enumerate(self.env.obstacles):
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = rospy.Time.now()
            m.ns = "obstacles"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(obs.center[0])
            m.pose.position.y = float(obs.center[1])
            m.pose.position.z = float(obs.center[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = obs.radius * 2.0
            m.color.r, m.color.g, m.color.b, m.color.a = 0.9, 0.2, 0.2, 0.5
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    def _joint_state_cb(self, msg: JointState):
        if self.joint_index_map is None:
            try:
                self.joint_index_map = [msg.name.index(j) for j in JOINT_NAMES]
            except ValueError:
                # joint_states hasn't published all 6 arm joints yet (e.g.
                # gripper-only message on a shared topic); wait for a
                # message that has them all.
                return
        self.current_q = np.array(
            [msg.position[idx] for idx in self.joint_index_map]
        )

    def _goal_pose_cb(self, msg: PoseStamped):
        if self.current_q is None:
            rospy.logwarn(
                "Received goal pose but no /joint_states received yet; ignoring."
            )
            return

        T_goal = pose_msg_to_matrix(msg)
        rospy.loginfo(
            f"Planning from current joints {np.round(np.rad2deg(self.current_q), 1)} "
            f"to goal position {np.round(T_goal[:3, 3], 3)} ..."
        )

        try:
            times, positions, velocities, accelerations, q_goal = plan_path(
                self.current_q,
                T_goal,
                self.env,
                duration=self.duration,
                n_samples=self.n_samples,
                max_bend_iterations=self.max_bend_iterations,
            )
        except TrajectoryPlanningError as e:
            rospy.logerr(f"Trajectory planning FAILED: {e}")
            return

        self._publish_trajectory(times, positions, velocities)
        rospy.loginfo(
            f"Published collision-free trajectory with {len(times)} waypoints "
            f"over {self.duration:.1f}s, ending at "
            f"{np.round(np.rad2deg(q_goal), 1)} deg."
        )

    def _publish_trajectory(self, times, positions, velocities):
        traj = JointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = JOINT_NAMES

        for t, pos, vel in zip(times, positions, velocities):
            pt = JointTrajectoryPoint()
            pt.positions = pos.tolist()
            pt.velocities = vel.tolist()
            pt.time_from_start = rospy.Duration.from_sec(float(t))
            traj.points.append(pt)

        self.traj_pub.publish(traj)


if __name__ == "__main__":
    try:
        node = UR5IKPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python3
"""
test_kinematics.py
===================
Automated regression tests for the UR5 kinematics, collision checking,
and trajectory planning modules. Run with:

    cd src/ur5_ik_planner/src/ur5_ik_planner && python3 -m pytest ../../test/test_kinematics.py -v

or, inside a catkin workspace:

    catkin run_tests ur5_ik_planner
"""

import sys
import os
import numpy as np
import pytest

# Make the package importable when run standalone (outside catkin)
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "src", "ur5_ik_planner")
)

from kinematics import (
    forward_kinematics, inverse_kinematics, filter_reachable, within_limits
)
from collision import Environment, check_joint_positions, check_trajectory
from trajectory import plan_path, joint_space_quintic_trajectory, TrajectoryPlanningError


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

class TestForwardKinematics:
    def test_zero_configuration_runs(self):
        T, joints = forward_kinematics(np.zeros(6))
        assert T.shape == (4, 4)
        assert len(joints) == 6

    def test_returns_valid_homogeneous_transform(self):
        q = np.deg2rad([10, -30, 60, -45, 20, 5])
        T, _ = forward_kinematics(q)
        # rotation part should be orthonormal
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
        # bottom row should be [0,0,0,1]
        assert np.allclose(T[3, :], [0, 0, 0, 1])


class TestInverseKinematics:
    def test_round_trip_recovers_a_valid_solution(self):
        """For 200 random configurations, IK must find at least one
        solution whose FK reproduces the original pose to high precision."""
        rng = np.random.RandomState(42)
        n_trials = 200
        failures = 0
        for _ in range(n_trials):
            q_true = rng.uniform(-np.pi, np.pi, 6)
            T_true, _ = forward_kinematics(q_true)
            sols = inverse_kinematics(T_true, q_current=np.zeros(6))
            if not sols:
                failures += 1
                continue
            T_check, _ = forward_kinematics(sols[0])
            assert np.allclose(T_check, T_true, atol=1e-6)
        # a handful of near-singular / boundary configurations may
        # legitimately fail; demand at least 95% success
        assert failures <= n_trials * 0.05, f"{failures}/{n_trials} IK failures"

    def test_known_pose_recovers_original_joint_angles(self):
        """A well-conditioned demo pose should have the exact original
        joint values among its solutions."""
        q_demo = np.deg2rad([20, -60, 90, -30, 45, 10])
        T_demo, _ = forward_kinematics(q_demo)
        sols = inverse_kinematics(T_demo, q_current=np.zeros(6))
        assert len(sols) >= 1
        assert any(np.allclose(s, q_demo, atol=1e-4) for s in sols)

    def test_returns_up_to_eight_solutions(self):
        q_demo = np.deg2rad([20, -60, 90, -30, 45, 10])
        T_demo, _ = forward_kinematics(q_demo)
        sols = inverse_kinematics(T_demo)
        assert 1 <= len(sols) <= 8

    def test_unreachable_pose_returns_empty(self):
        # a pose far beyond the UR5's maximum reach
        T_far = np.eye(4)
        T_far[:3, 3] = [5.0, 5.0, 5.0]
        sols = inverse_kinematics(T_far)
        assert sols == []

    def test_solutions_are_all_distinct(self):
        q_demo = np.deg2rad([20, -60, 90, -30, 45, 10])
        T_demo, _ = forward_kinematics(q_demo)
        sols = inverse_kinematics(T_demo)
        for i in range(len(sols)):
            for j in range(i + 1, len(sols)):
                assert not np.allclose(sols[i], sols[j], atol=1e-3)


class TestJointLimits:
    def test_within_limits_basic(self):
        limits = [(-1.0, 1.0)] * 6
        assert within_limits([0, 0, 0, 0, 0, 0], limits)
        assert not within_limits([2.0, 0, 0, 0, 0, 0], limits)

    def test_filter_reachable_drops_out_of_range(self):
        limits = [(-1.0, 1.0)] * 6
        sols = [np.zeros(6), np.array([2.0, 0, 0, 0, 0, 0])]
        filtered = filter_reachable(sols, limits)
        assert len(filtered) == 1


# ---------------------------------------------------------------------------
# Collision checking
# ---------------------------------------------------------------------------

class TestCollision:
    def test_no_obstacles_never_collides(self):
        env = Environment()
        _, joints = forward_kinematics(np.deg2rad([0, -90, 90, 0, 90, 0]))
        collided, details = check_joint_positions(joints, env)
        assert not collided
        assert details == []

    def test_obstacle_directly_on_link_detected(self):
        q = np.deg2rad([0, -90, 0, 0, 0, 0])
        _, joints = forward_kinematics(q)
        env = Environment()
        env.add(center=joints[2], radius=0.15, name="direct_hit")
        collided, details = check_joint_positions(joints, env)
        assert collided
        assert len(details) > 0

    def test_distant_obstacle_not_detected(self):
        q = np.deg2rad([0, -90, 0, 0, 0, 0])
        _, joints = forward_kinematics(q)
        env = Environment()
        env.add(center=[10, 10, 10], radius=0.1, name="far_away")
        collided, _ = check_joint_positions(joints, env)
        assert not collided

    def test_check_trajectory_finds_first_collision(self):
        q_start = np.deg2rad([0, -90, 90, 0, 90, 0])
        q_goal = np.deg2rad([30, -90, 90, 0, 90, 0])
        _, samples_pos = forward_kinematics(q_start)
        env = Environment()
        env.add(center=samples_pos[2], radius=0.2, name="blocker")
        traj = [q_start, q_goal]
        is_clear, idx, details = check_trajectory(traj, env, forward_kinematics)
        assert not is_clear
        assert idx == 0


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

class TestQuinticTrajectory:
    def test_boundary_conditions_zero_velocity(self):
        q_start = np.deg2rad([0, -90, 90, 0, 90, 0])
        q_goal = np.deg2rad([30, -60, 60, -20, 45, 10])
        times, pos, vel, acc = joint_space_quintic_trajectory(
            q_start, q_goal, duration=3.0, n_samples=50
        )
        assert np.allclose(vel[0], 0, atol=1e-9)
        assert np.allclose(vel[-1], 0, atol=1e-6)
        assert np.allclose(acc[0], 0, atol=1e-9)
        assert np.allclose(acc[-1], 0, atol=1e-6)

    def test_endpoints_match_start_and_goal(self):
        q_start = np.deg2rad([0, -90, 90, 0, 90, 0])
        q_goal = np.deg2rad([30, -60, 60, -20, 45, 10])
        times, pos, vel, acc = joint_space_quintic_trajectory(
            q_start, q_goal, duration=3.0, n_samples=50
        )
        assert np.allclose(pos[0], q_start, atol=1e-9)
        assert np.allclose(pos[-1], q_goal, atol=1e-6)

    def test_monotonic_time_samples(self):
        q_start = np.zeros(6)
        q_goal = np.deg2rad([10, 10, 10, 10, 10, 10])
        times, *_ = joint_space_quintic_trajectory(q_start, q_goal, 2.0, n_samples=30)
        assert np.all(np.diff(times) > 0)


class TestPlanPath:
    def test_unobstructed_path_succeeds(self):
        q_start = np.deg2rad([0, -90, 90, 0, 90, 0])
        T_start, _ = forward_kinematics(q_start)
        T_goal = T_start.copy()
        T_goal[:3, 3] += np.array([0.1, 0.05, -0.05])
        env = Environment()  # no obstacles

        times, pos, vel, acc, q_goal = plan_path(
            q_start, T_goal, env, duration=3.0, n_samples=50
        )
        assert len(times) == 50
        is_clear, _, _ = check_trajectory(pos, env, forward_kinematics)
        assert is_clear

    def test_unreachable_goal_raises(self):
        q_start = np.deg2rad([0, -90, 90, 0, 90, 0])
        T_far = np.eye(4)
        T_far[:3, 3] = [5.0, 5.0, 5.0]
        env = Environment()
        with pytest.raises(TrajectoryPlanningError):
            plan_path(q_start, T_far, env, duration=3.0, n_samples=30)

    def test_planner_never_returns_a_colliding_trajectory(self):
        """Critical safety property: across many random obstacle
        configurations, plan_path must either return a verified clear
        trajectory or raise -- it must never silently return a
        trajectory that collides."""
        unsafe = 0
        n_trials = 20
        for trial in range(n_trials):
            rng = np.random.RandomState(trial + 500)
            q_start = np.deg2rad(rng.uniform(-60, 60, 6))
            q_start[1] = np.deg2rad(rng.uniform(-110, -40))
            T_start, _ = forward_kinematics(q_start)

            q_goal_seed = np.deg2rad(rng.uniform(-60, 60, 6))
            q_goal_seed[1] = np.deg2rad(rng.uniform(-110, -40))
            T_goal, _ = forward_kinematics(q_goal_seed)

            mid = (T_start[:3, 3] + T_goal[:3, 3]) / 2
            env = Environment()
            env.add(center=mid, radius=0.04, name="blocker")

            try:
                _, pos, _, _, _ = plan_path(
                    q_start, T_goal, env, duration=3.0, n_samples=60,
                    max_bend_iterations=8
                )
                is_clear, _, _ = check_trajectory(pos, env, forward_kinematics)
                if not is_clear:
                    unsafe += 1
            except TrajectoryPlanningError:
                pass  # expected/safe: correctly refused

        assert unsafe == 0, f"{unsafe}/{n_trials} unsafe trajectories returned!"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

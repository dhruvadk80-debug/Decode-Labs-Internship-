#!/usr/bin/env python3
"""
trajectory.py
=============
Smooth, collision-free joint-space trajectory generation for the UR5,
connecting a start pose (Point A) to a goal pose (Point B).

Pipeline
--------
1. Solve IK for the goal pose, filter by joint limits, and pick the
   solution closest to the current joint configuration (minimizes
   unnecessary motion / joint flips).
2. Generate a smooth trajectory in JOINT SPACE between q_start and
   q_goal using a quintic (5th-order) polynomial per joint. Quintic
   splines give zero velocity AND zero acceleration at both endpoints,
   which is what you want for a clean start/stop motion (no jerk at
   the boundaries).
3. Sample the spline densely and run it through the collision checker
   (collision.py). If a collision is found, the trajectory is "bent"
   away from the obstacle using a simple via-point repulsion heuristic
   and re-checked; this repeats up to a maximum number of iterations.

This is a deliberately simple, dependency-light planner (no MoveIt/OMPL
required) appropriate for the scope of "Point A to Point B, collision-
free, smooth spline" in this project. For complex cluttered scenes, a
sampling-based planner (RRT*, etc., e.g. via MoveIt/OMPL in the real
ROS stack) would be more robust -- swap implementations behind the
same plan_path() interface if you outgrow this.
"""

import numpy as np

from kinematics import forward_kinematics, inverse_kinematics, filter_reachable, JOINT_LIMITS
from collision import Environment, check_trajectory, check_joint_positions


class TrajectoryPlanningError(Exception):
    """Raised when no collision-free trajectory could be found."""
    pass


def _quintic_coeffs(q0, qf, T, v0=0.0, vf=0.0, a0=0.0, af=0.0):
    """
    Coefficients of a quintic polynomial q(t) = c0 + c1 t + ... + c5 t^5
    over t in [0, T], matching given boundary position/velocity/
    acceleration at both ends. Returns the 6 coefficients (low to high).
    """
    c0 = q0
    c1 = v0
    c2 = a0 / 2.0
    # solve remaining 3 coefficients from the 3 end-boundary conditions
    M = np.array([
        [T ** 3,      T ** 4,       T ** 5],
        [3 * T ** 2,  4 * T ** 3,   5 * T ** 4],
        [6 * T,       12 * T ** 2,  20 * T ** 3],
    ])
    rhs = np.array([
        qf - (c0 + c1 * T + c2 * T ** 2),
        vf - (c1 + 2 * c2 * T),
        af - (2 * c2),
    ])
    c3, c4, c5 = np.linalg.solve(M, rhs)
    return np.array([c0, c1, c2, c3, c4, c5])


def _quintic_eval(coeffs, t):
    """Evaluate position, velocity, acceleration of a quintic at time t."""
    c0, c1, c2, c3, c4, c5 = coeffs
    pos = c0 + c1 * t + c2 * t ** 2 + c3 * t ** 3 + c4 * t ** 4 + c5 * t ** 5
    vel = c1 + 2 * c2 * t + 3 * c3 * t ** 2 + 4 * c4 * t ** 3 + 5 * c5 * t ** 4
    acc = 2 * c2 + 6 * c3 * t + 12 * c4 * t ** 2 + 20 * c5 * t ** 3
    return pos, vel, acc


def joint_space_quintic_trajectory(q_start, q_goal, duration, n_samples=100,
                                    via_points=None):
    """
    Generate a smooth quintic joint-space trajectory from q_start to
    q_goal (with zero velocity/acceleration at both ends), optionally
    passing through intermediate via-points (used to bend the path
    around obstacles).

    Parameters
    ----------
    q_start, q_goal : array-like(6,)
        Start and goal joint configurations, radians.
    duration : float
        Total trajectory time, seconds.
    n_samples : int
        Number of waypoints to sample along the trajectory.
    via_points : list of (t_fraction, q) tuples, optional
        Intermediate joint configurations to pass through, where
        t_fraction in (0, 1) is the fraction of `duration` at which to
        pass through q. If given, the trajectory is built as a sequence
        of quintic segments through start -> via points -> goal, with
        matched (zero) velocity/acceleration only at the overall start
        and end (via points are positionally interpolated with simple
        velocity continuity).

    Returns
    -------
    times : np.ndarray(n_samples,)
    positions : np.ndarray(n_samples, 6)
    velocities : np.ndarray(n_samples, 6)
    accelerations : np.ndarray(n_samples, 6)
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    if not via_points:
        coeffs_per_joint = [
            _quintic_coeffs(q_start[j], q_goal[j], duration) for j in range(6)
        ]
        times = np.linspace(0, duration, n_samples)
        positions = np.zeros((n_samples, 6))
        velocities = np.zeros((n_samples, 6))
        accelerations = np.zeros((n_samples, 6))
        for j in range(6):
            for i, t in enumerate(times):
                pos, vel, acc = _quintic_eval(coeffs_per_joint[j], t)
                positions[i, j] = pos
                velocities[i, j] = vel
                accelerations[i, j] = acc
        return times, positions, velocities, accelerations

    # With via points: build a waypoint list (time, q) and fit a
    # quintic segment between each consecutive pair, with zero
    # vel/accel only at the very first and very last waypoint, and
    # numerically-estimated (finite-difference) velocity continuity at
    # interior via points.
    waypoint_times = [0.0] + [t * duration for t, _ in via_points] + [duration]
    waypoint_qs = [q_start] + [np.asarray(q) for _, q in via_points] + [q_goal]

    n_segs = len(waypoint_times) - 1
    # estimate interior velocities via central differences (Catmull-Rom-like)
    interior_vels = []
    for k in range(1, len(waypoint_qs) - 1):
        dt = waypoint_times[k + 1] - waypoint_times[k - 1]
        v = (waypoint_qs[k + 1] - waypoint_qs[k - 1]) / dt if dt > 1e-9 else np.zeros(6)
        interior_vels.append(v)
    vels = [np.zeros(6)] + interior_vels + [np.zeros(6)]

    times = np.linspace(0, duration, n_samples)
    positions = np.zeros((n_samples, 6))
    velocities = np.zeros((n_samples, 6))
    accelerations = np.zeros((n_samples, 6))

    seg_coeffs = []
    for s in range(n_segs):
        T_seg = waypoint_times[s + 1] - waypoint_times[s]
        coeffs_j = [
            _quintic_coeffs(waypoint_qs[s][j], waypoint_qs[s + 1][j], T_seg,
                             v0=vels[s][j], vf=vels[s + 1][j])
            for j in range(6)
        ]
        seg_coeffs.append((waypoint_times[s], T_seg, coeffs_j))

    for i, t in enumerate(times):
        # find which segment t falls in
        seg = 0
        for s, (t0, T_seg, _) in enumerate(seg_coeffs):
            if t >= t0 - 1e-9:
                seg = s
        t0, T_seg, coeffs_j = seg_coeffs[seg]
        t_local = np.clip(t - t0, 0.0, T_seg)
        for j in range(6):
            pos, vel, acc = _quintic_eval(coeffs_j[j], t_local)
            positions[i, j] = pos
            velocities[i, j] = vel
            accelerations[i, j] = acc

    return times, positions, velocities, accelerations


def _bend_via_point(q_base, obstacle_center, fk_func, push_distance=0.25):
    """
    Construct a single via-point joint configuration that nudges the
    path away from an obstacle. Strategy: take the given base joint
    configuration (typically the colliding waypoint), compute its
    Cartesian tool position, push that position away from the obstacle
    center by push_distance, then IK back to joint space. If the pushed
    target is unreachable, progressively shrink the push distance
    before giving up (overshooting the workspace is a common failure
    mode for large pushes near the arm's reach limit).
    """
    q_base = np.asarray(q_base)
    T_base, _ = fk_func(q_base)
    p_base = T_base[:3, 3]

    direction = p_base - np.asarray(obstacle_center)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([0.0, 0.0, 1.0])
        norm = 1.0
    direction = direction / norm

    for scale in (1.0, 0.6, 0.35, 0.2):
        p_pushed = p_base + direction * push_distance * scale
        T_pushed = T_base.copy()
        T_pushed[:3, 3] = p_pushed

        sols = inverse_kinematics(T_pushed, q_current=q_base)
        sols = filter_reachable(sols)
        if sols:
            return sols[0]

    return None


def plan_path(q_start, goal_pose, env: Environment, duration=4.0, n_samples=100,
              max_bend_iterations=5, joint_limits=JOINT_LIMITS):
    """
    Plan a smooth, collision-free joint-space trajectory from the
    current joint configuration to a Cartesian goal pose.

    Parameters
    ----------
    q_start : array-like(6,)
        Current joint configuration, radians.
    goal_pose : 4x4 array-like
        Desired tool pose (Point B) in the base frame.
    env : Environment
        Obstacles to avoid.
    duration : float
        Trajectory duration, seconds.
    n_samples : int
        Number of waypoints to sample/check along the trajectory.
    max_bend_iterations : int
        Maximum number of via-point "bend away from obstacle" attempts
        before giving up.
    joint_limits : list of (min, max) tuples
        Per-joint angle limits used to filter IK solutions.

    Returns
    -------
    times, positions, velocities, accelerations : as returned by
        joint_space_quintic_trajectory.
    q_goal : np.ndarray(6,)
        The joint configuration used for the goal (the IK solution
        chosen).

    Raises
    ------
    TrajectoryPlanningError
        If the goal pose is unreachable, or no collision-free
        trajectory could be found within max_bend_iterations attempts.
    """
    q_start = np.asarray(q_start, dtype=float)

    sols = inverse_kinematics(goal_pose, q_current=q_start)
    sols = filter_reachable(sols, joint_limits)
    if not sols:
        raise TrajectoryPlanningError(
            "Goal pose is unreachable: no IK solution within joint limits."
        )
    q_goal = sols[0]

    via_points = []
    for attempt in range(max_bend_iterations + 1):
        times, positions, velocities, accelerations = joint_space_quintic_trajectory(
            q_start, q_goal, duration, n_samples=n_samples, via_points=via_points or None
        )

        is_clear, collide_idx, details = check_trajectory(
            positions, env, forward_kinematics
        )
        if is_clear:
            return times, positions, velocities, accelerations, q_goal

        if attempt == max_bend_iterations:
            raise TrajectoryPlanningError(
                f"Could not find a collision-free trajectory after "
                f"{max_bend_iterations} bend attempts. Last collision at "
                f"waypoint {collide_idx}: {details}"
            )

        # bend away from the first obstacle implicated in the collision
        offending_obstacle = env.obstacles[0] if env.obstacles else None
        if offending_obstacle is None:
            raise TrajectoryPlanningError(
                "Trajectory collides but no obstacle is registered in the "
                "environment (likely a ground-plane violation)."
            )

        via_q = _bend_via_point(
            positions[collide_idx], offending_obstacle.center, forward_kinematics,
            push_distance=min(0.15 + 0.05 * attempt, 0.35)
        )
        if via_q is None:
            raise TrajectoryPlanningError(
                "Could not compute an IK-reachable via-point to avoid the obstacle."
            )

        t_fraction = (collide_idx / max(n_samples - 1, 1))
        t_fraction = float(np.clip(t_fraction, 0.15, 0.85))
        via_points = [(t_fraction, via_q)]

    # unreachable in practice (loop always returns or raises above)
    raise TrajectoryPlanningError("Trajectory planning failed unexpectedly.")

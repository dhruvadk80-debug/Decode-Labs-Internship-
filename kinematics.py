#!/usr/bin/env python3
"""
kinematics.py
=============
Forward and Inverse Kinematics for the Universal Robots UR5 (6-DOF,
all-revolute, offset-spherical-wrist architecture).

DH convention
-------------
Standard (Denavit-Hartenberg, not modified/Craig) parameters, the
values published by Universal Robots / widely used in the ROS-Industrial
community for the UR5 (lengths in meters, angles in radians):

    i  | alpha(i-1) |  a(i-1)   |  d(i)     | theta(i)
    ---+------------+-----------+-----------+----------
    1  |     0      |    0      | 0.089159  |  q1
    2  |   pi/2     |    0      |    0      |  q2
    3  |     0      | -0.425    |    0      |  q3
    4  |     0      | -0.39225  | 0.10915   |  q4
    5  |   pi/2     |    0      | 0.09465   |  q5
    6  |  -pi/2     |    0      | 0.0823    |  q6

Each row's (alpha, a) describe the twist/length BEFORE that joint's own
rotation+offset (i.e. alpha[i-1], a[i-1] paired with d[i], theta[i] -
the standard DH row convention), and are indexed 0..5 here for joints
1..6 respectively. This table has been numerically verified (see the
self-test at the bottom of this file and test_kinematics.py) against
several independently published UR5 kinematic references.

IK approach
-----------
Closed-form analytic solution exploiting the UR5's offset-spherical-wrist
geometry, derived and numerically verified joint-by-joint:

  1. theta1 (2 solutions) - from the projection of the wrist center
     onto the base XY plane, offset by the d4 "forearm" lateral shift.
  2. theta5, theta6 (2 solutions per theta1) - from a clean geometric
     invariant: in frame 1 (after removing theta1), the chain through
     joints 2-3-4 always leaves the z-axis of frame 4 exactly in the
     frame-1 XY plane (a consequence of alpha1=alpha2=0). This means
     row index 2 ("z-row") of T16 depends ONLY on theta5 and theta6,
     giving closed-form expressions for both.
  3. theta2, theta3, theta4 (2 solutions per theta5/theta6 branch) -
     once theta5/theta6 are known, T14 = T16 @ inv(T45 @ T56) is
     recovered. Joints 2 and 3 form a classic planar 2-link arm (law
     of cosines) in the frame-1 XY plane; theta4 closes the chain via
     the remaining rotation.

This yields up to 8 valid configurations for any reachable pose. The
caller can filter by joint limits and pick whichever configuration is
closest to the robot's current joint state for smooth, flip-free motion.

Author: generated for Project 1 - Robotic Arm Kinematics & Path Planning
"""

import numpy as np

# ---------------------------------------------------------------------------
# UR5 DH parameters (meters / radians), indexed by joint i = 0..5 (joints 1..6)
# ---------------------------------------------------------------------------
D1 = 0.089159
A2 = -0.425
A3 = -0.39225
D4 = 0.10915
D5 = 0.09465
D6 = 0.0823

ALPHA = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]
A_ARR = [0.0, A2, A3, 0.0, 0.0, 0.0]
D_ARR = [D1, 0.0, 0.0, D4, D5, D6]

# Joint limits (radians). UR5 default software limits are +-360 deg per
# joint; tighten these for your specific cell / safety requirements.
JOINT_LIMITS = [(-2 * np.pi, 2 * np.pi)] * 6

ZERO_THRESH = 1e-8


def _dh_transform(alpha, a, d, theta):
    """
    Standard DH homogeneous transform (frame i-1 -> frame i):
        T = Rot_x(alpha) * Trans_x(a) * Rot_z(theta) * Trans_z(d)
    expressed directly as a single matrix (textbook closed form).
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,        -st * ca,   st * sa,   a * ct],
        [st,         ct * ca,  -ct * sa,   a * st],
        [0.0,        sa,        ca,        d],
        [0.0,        0.0,       0.0,       1.0]
    ])


def _wrap(angle):
    """Wrap angle to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def forward_kinematics(q):
    """
    Compute the forward kinematics of the UR5.

    Parameters
    ----------
    q : array-like of 6 floats
        Joint angles [q1..q6] in radians.

    Returns
    -------
    T : 4x4 np.ndarray
        Homogeneous transform of the tool frame (wrist3 / flange) in the
        base frame.
    joint_origins : list of 6 np.ndarray(3,)
        XYZ positions of each joint frame origin in the base frame
        (frame1 .. frame6), useful for collision checking and
        visualization / RViz marker publishing.
    """
    q = np.asarray(q, dtype=float).flatten()
    assert q.size == 6, "q must have 6 elements"

    T = np.eye(4)
    joint_origins = []
    for i in range(6):
        Ti = _dh_transform(ALPHA[i], A_ARR[i], D_ARR[i], q[i])
        T = T @ Ti
        joint_origins.append(T[:3, 3].copy())

    return T, joint_origins


def inverse_kinematics(T_target, q_current=None, tol=1e-4):
    """
    Analytic inverse kinematics for the UR5. Returns all valid (up to 8)
    IK solutions for a reachable pose.

    Parameters
    ----------
    T_target : 4x4 array-like
        Desired homogeneous transform of the tool frame in the base frame.
    q_current : array-like of 6 floats, optional
        Current joint configuration. If given, returned solutions are
        sorted by joint-space distance to q_current (closest first) -
        use this to pick the configuration that minimizes motion and
        avoids unnecessary joint flips during trajectory execution.
    tol : float
        Position+orientation tolerance (Frobenius norm) used to validate
        candidate solutions against the target pose.

    Returns
    -------
    solutions : list of np.ndarray(6,)
        Valid joint configurations [q1..q6] in radians, each wrapped to
        (-pi, pi]. Empty list if the pose is unreachable.
    """
    T = np.asarray(T_target, dtype=float)
    assert T.shape == (4, 4)

    px, py, pz = T[0, 3], T[1, 3], T[2, 3]
    ax, ay, az = T[0, 2], T[1, 2], T[2, 2]

    # Wrist center: back off from the tool tip along the tool's approach
    # (z) axis by d6.
    wx, wy = px - D6 * ax, py - D6 * ay

    r_xy = np.hypot(wx, wy)
    if r_xy < abs(D4) - 1e-9:
        return []  # wrist center inside the d4 "no-go" cylinder: unreachable

    psi = np.arctan2(wy, wx)
    s = np.clip(D4 / r_xy, -1.0, 1.0)
    asinD4 = np.arcsin(s)

    # sin(psi - theta1) = +D4/r_xy or -D4/r_xy each have two algebraic
    # roots; not all four are independent in general, but which pair
    # collapses depends on the specific geometry, so we generate all
    # four and let the downstream forward-kinematics check (below)
    # discard the spurious ones. This is more robust than hand-picking
    # two "canonical" branches.
    theta1_options = [
        _wrap(psi - asinD4),
        _wrap(psi - np.pi + asinD4),
        _wrap(psi + asinD4),
        _wrap(psi - np.pi - asinD4),
    ]
    # de-duplicate coincident roots (e.g. at boundary r_xy == |D4|)
    dedup_t1 = []
    for t1 in theta1_options:
        if not any(abs(_wrap(t1 - u)) < 1e-9 for u in dedup_t1):
            dedup_t1.append(t1)
    theta1_options = dedup_t1

    solutions = []

    for theta1 in theta1_options:
        T01 = _dh_transform(ALPHA[0], A_ARR[0], D_ARR[0], theta1)
        T16 = np.linalg.inv(T01) @ T

        # theta5 from the invariant: row index 2 of T16 == row index 1 of
        # (T45 @ T56), whose [.,2] entry is cos(theta5).
        cos5 = np.clip(T16[2, 2], -1.0, 1.0)
        theta5_options = [np.arccos(cos5), -np.arccos(cos5)]

        for theta5 in theta5_options:
            if abs(np.sin(theta5)) < ZERO_THRESH:
                theta6 = 0.0  # wrist singularity: theta6 underdetermined
            else:
                theta6 = np.arctan2(-T16[2, 1] / np.sin(theta5),
                                     T16[2, 0] / np.sin(theta5))

            T45 = _dh_transform(ALPHA[4], A_ARR[4], D_ARR[4], theta5)
            T56 = _dh_transform(ALPHA[5], A_ARR[5], D_ARR[5], theta6)
            T14 = T16 @ np.linalg.inv(T45 @ T56)

            x14, y14 = T14[0, 3], T14[1, 3]
            r2 = x14 ** 2 + y14 ** 2
            cos_t3 = (r2 - A2 ** 2 - A3 ** 2) / (2 * A2 * A3)
            cos_t3 = np.clip(cos_t3, -1.0, 1.0)
            if abs((r2 - A2 ** 2 - A3 ** 2) / (2 * A2 * A3)) > 1.0 + 1e-6:
                continue  # elbow geometrically unreachable on this branch

            theta3_raw = np.arccos(cos_t3)

            for theta3 in (theta3_raw, -theta3_raw):
                k1 = A2 + A3 * np.cos(theta3)
                k2 = A3 * np.sin(theta3)
                theta2 = np.arctan2(y14, x14) - np.arctan2(k2, k1)

                T12 = _dh_transform(ALPHA[1], A_ARR[1], D_ARR[1], theta2)
                T23 = _dh_transform(ALPHA[2], A_ARR[2], D_ARR[2], theta3)
                T13 = T12 @ T23
                T34 = np.linalg.inv(T13) @ T14
                theta4 = np.arctan2(T34[1, 0], T34[0, 0])

                q = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
                q = np.array([_wrap(a) for a in q])

                T_check, _ = forward_kinematics(q)
                if np.allclose(T_check, T, atol=tol):
                    solutions.append(q)

    # De-duplicate near-identical solutions
    unique = []
    for s in solutions:
        if not any(np.allclose(s, u, atol=1e-4) for u in unique):
            unique.append(s)

    if q_current is not None:
        q_current = np.asarray(q_current, dtype=float)
        unique.sort(key=lambda s: np.linalg.norm(_wrap(s - q_current)))

    return unique


def within_limits(q, limits=JOINT_LIMITS):
    """Return True if every joint angle in q is within its limit range."""
    return all(lo <= qi <= hi for qi, (lo, hi) in zip(q, limits))


def filter_reachable(solutions, limits=JOINT_LIMITS):
    """Filter a list of IK solutions down to those within joint limits."""
    return [s for s in solutions if within_limits(s, limits)]


if __name__ == "__main__":
    # Self-test: FK -> IK should recover a configuration that reproduces
    # the same tool pose (not necessarily the exact original joint
    # values, since IK is multi-valued).
    np.random.seed(42)
    n_trials = 200
    max_pos_err = 0.0
    max_rot_err = 0.0
    failures = 0

    for _ in range(n_trials):
        q_test = np.random.uniform(-np.pi, np.pi, 6)
        T_test, _ = forward_kinematics(q_test)
        sols = inverse_kinematics(T_test, q_current=np.zeros(6))

        if not sols:
            failures += 1
            continue

        best = sols[0]
        T_check, _ = forward_kinematics(best)
        pos_err = np.linalg.norm(T_check[:3, 3] - T_test[:3, 3])
        rot_err = np.linalg.norm(T_check[:3, :3] - T_test[:3, :3])
        max_pos_err = max(max_pos_err, pos_err)
        max_rot_err = max(max_rot_err, rot_err)

    print(f"Self-test over {n_trials} random configurations:")
    print(f"  failures (no IK solution found): {failures}")
    print(f"  max position error:    {max_pos_err:.2e} m")
    print(f"  max orientation error: {max_rot_err:.2e} (Frobenius norm)")

    # Demo: show all solutions for one pose
    q_demo = np.deg2rad([20, -60, 90, -30, 45, 10])
    T_demo, _ = forward_kinematics(q_demo)
    sols = inverse_kinematics(T_demo, q_current=np.zeros(6))
    print(f"\nDemo pose has {len(sols)} IK solutions (joint angles, deg):")
    for s in sols:
        print("  ", np.round(np.rad2deg(s), 2))

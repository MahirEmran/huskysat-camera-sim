import numpy as np
from Basilisk.utilities import RigidBodyKinematics as rbk


EARTH_RADIUS_M = 6371.0e3


def angle_deg_between(vec_a, vec_b):
    """Return angle in degrees between two vectors."""
    # clip protects arccos from tiny floating-point overshoot beyond [-1, 1]
    return np.degrees(np.arccos(np.clip(np.dot(vec_a, vec_b), -1.0, 1.0)))


def safe_unit(vec):
    """Return normalized vector or None if magnitude is too small."""
    mag = np.linalg.norm(vec)
    if mag < 1e-12:
        return None
    return vec / mag


def compute_camera_uptime_flags(
    sc_state,
    sun_state,
    moon_state,
    vec_lost_b,
    vec_found_b,
    pos_lost_b,
    pos_found_b,
    lost_inner_keepout_half_deg,
    found_earth_keepin_half_deg,
    found_sun_keepout_half_deg,
):
    """Evaluate LOST and FOUND camera constraint satisfaction at one sample instant."""
    # Pull current translational + attitude state from Basilisk message payload.
    r_N = np.array(sc_state.r_BN_N, dtype=float)
    sigma_BN = np.array(sc_state.sigma_BN, dtype=float)

    # If state is clearly invalid, don't count this instant as successful uptime.
    if np.linalg.norm(r_N) < 1.0 or sigma_BN.size != 3:
        return False, False

    # c_bn maps inertial -> body, so transpose gives body -> inertial.
    # We need body->inertial to express boresight axes in inertial space.
    c_bn = rbk.MRP2C(sigma_BN)
    c_nb = c_bn.T

    lost_hat_N = safe_unit(c_nb.dot(np.array(vec_lost_b, dtype=float)))
    found_hat_N = safe_unit(c_nb.dot(np.array(vec_found_b, dtype=float)))
    if lost_hat_N is None or found_hat_N is None:
        return False, False

    # Earth is at the inertial origin in this simulation setup.
    earth_pos_N = np.zeros(3)
    sun_pos_N = np.array(sun_state.PositionVector, dtype=float)
    moon_pos_N = np.array(moon_state.PositionVector, dtype=float)

    lost_pos_N = r_N + c_nb.dot(np.array(pos_lost_b, dtype=float))
    found_pos_N = r_N + c_nb.dot(np.array(pos_found_b, dtype=float))

    # For LOST we evaluate three potentially bright/blocked bodies.
    lost_dirs = [
        safe_unit(earth_pos_N - lost_pos_N),
        safe_unit(sun_pos_N - lost_pos_N),
        safe_unit(moon_pos_N - lost_pos_N),
    ]
    found_earth_dir = safe_unit(earth_pos_N - found_pos_N)
    found_sun_dir = safe_unit(sun_pos_N - found_pos_N)

    if any(direction is None for direction in lost_dirs) or found_earth_dir is None or found_sun_dir is None:
        return False, False

    # LOST is "good" only if Earth/Sun/Moon are all outside the INNER (red) keep-out cone.
    # i.e., each angle from LOST boresight must be >= inner half-angle.
    lost_ok = all(
        angle_deg_between(lost_hat_N, direction) >= lost_inner_keepout_half_deg
        for direction in lost_dirs
    )

    found_earth_ang_deg = angle_deg_between(found_hat_N, found_earth_dir)
    found_sun_ang_deg = angle_deg_between(found_hat_N, found_sun_dir)
    # Earth occupies an angular disk as seen from the camera.
    # Treat FOUND as valid if the camera cone intersects that disk, not only if Earth center is inside.
    found_earth_range_m = np.linalg.norm(found_pos_N)
    if found_earth_range_m <= EARTH_RADIUS_M:
        earth_half_angle_deg = 90.0
    else:
        earth_half_angle_deg = np.degrees(np.arcsin(EARTH_RADIUS_M / found_earth_range_m))
    found_earth_visible = found_earth_ang_deg <= (found_earth_keepin_half_deg + earth_half_angle_deg)

    # FOUND is "good" when Earth is in-view and Sun is outside the INNER (red) keep-out cone.
    # This is a conjunction, not a score: both conditions must hold simultaneously.
    found_ok = (
        found_earth_visible
        and found_sun_ang_deg >= found_sun_keepout_half_deg
    )

    return lost_ok, found_ok

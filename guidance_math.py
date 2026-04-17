from datetime import datetime, timezone
import numpy as np


def approx_sun_hat_from_epoch(epoch_iso_utc):
    """Approximate inertial Earth->Sun unit vector from epoch (UTC ISO-8601)."""
    dt = datetime.fromisoformat(epoch_iso_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    # We only need a stable Sun direction, not deep ephemeris accuracy.
    # This low-order model is good enough for pointing decisions.
    # Days since J2000 epoch (2000-01-01T12:00:00Z).
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    n_days = (dt - j2000).total_seconds() / 86400.0
    # Mean longitude and mean anomaly in degrees, then converted to radians.
    # These are standard low-order solar terms used in quick ephemeris approximations.
    mean_long = np.radians((280.460 + 0.9856474 * n_days) % 360.0)
    mean_anom = np.radians((357.528 + 0.9856003 * n_days) % 360.0)
    # Correct the mean position to get apparent ecliptic longitude.
    lam = mean_long + np.radians(1.915) * np.sin(mean_anom) + np.radians(0.020) * np.sin(2.0 * mean_anom)
    eps = np.radians(23.439 - 0.0000004 * n_days)
    # Convert ecliptic (lam, eps) to inertial xyz and normalize.
    sun_hat = np.array([
        np.cos(lam),
        np.cos(eps) * np.sin(lam),
        np.sin(eps) * np.sin(lam),
    ])
    return sun_hat / np.linalg.norm(sun_hat)


def compute_compromise_x(earth_hat, sun_hat, rho_rad):
    """Build the EXPERIMENT-mode +X target on the Earth-limb cone away from the Sun."""
    # u1 is nadir. We want +X to stay on a cone around nadir with half-angle rho.
    u1 = earth_hat

    # Move away from the Sun by projecting anti-sun into nadir's perpendicular plane.
    # Projection formula: u2 = anti_sun - (anti_sun·u1)u1
    anti_sun = -sun_hat
    u2 = anti_sun - np.dot(anti_sun, u1) * u1
    n2 = np.linalg.norm(u2)

    # Degenerate case: if anti-sun is nearly collinear with nadir, pick any valid perpendicular.
    if n2 < 1e-6:
        u2 = np.array([1, 0, 0]) if abs(u1[0]) < 0.9 else np.array([0, 1, 0])
        u2 = u2 - np.dot(u2, u1) * u1
        n2 = np.linalg.norm(u2)
    u2 /= n2

    # x_b is the weighted blend on the cone:
    #   x_b = cos(rho) * u1 + sin(rho) * u2
    # This keeps +X at fixed Earth-limb angle rho while selecting anti-sun azimuth.
    x_b = np.cos(rho_rad) * u1 + np.sin(rho_rad) * u2
    return x_b / np.linalg.norm(x_b)


def build_yz_frame_about_x(x_b, earth_hat):
    """Create a stable right-handed frame around x_B with y/z used for roll sweeping."""
    # Build a temporary z axis by crossing x with nadir.
    # If they are close to parallel, fall back to other reference axes.
    z_temp = np.cross(x_b, earth_hat)
    if np.linalg.norm(z_temp) < 1e-6:
        z_temp = np.cross(x_b, np.array([0, 1, 0]))
        if np.linalg.norm(z_temp) < 1e-6:
            z_temp = np.cross(x_b, np.array([0, 0, 1]))
    z_temp /= np.linalg.norm(z_temp)
    # Complete right-handed triad: y = z x x.
    y_temp = np.cross(z_temp, x_b)
    y_temp /= np.linalg.norm(y_temp)
    return y_temp, z_temp


def solve_roll_for_lost_clearance(x_b, earth_hat, sun_hat, prev_roll_deg=None):
    """Choose roll that maximizes the minimum LOST (+Z) clearance from Earth and Sun."""
    y_temp, z_temp = build_yz_frame_about_x(x_b, earth_hat)

    roll_candidates = []
    max_score = -1.0

    # Sweep roll in 2 deg increments.
    # For each roll, LOST (+Z) gets a different inertial direction.
    # Score = min(angle_to_earth, angle_to_sun), so maximizing score protects worst case.
    for roll_deg in range(0, 360, 2):
        roll = np.radians(roll_deg)
        y_test = np.cos(roll) * y_temp + np.sin(roll) * z_temp
        z_test = np.cross(x_b, y_test)
        z_test /= np.linalg.norm(z_test)
        lost_hat = z_test

        ang_earth = np.degrees(np.arccos(np.clip(np.dot(lost_hat, earth_hat), -1.0, 1.0)))
        ang_sun = np.degrees(np.arccos(np.clip(np.dot(lost_hat, sun_hat), -1.0, 1.0)))
        score = min(ang_earth, ang_sun)

        roll_candidates.append((roll_deg, score, y_test, z_test))
        if score > max_score:
            max_score = score

    # Keep all options that are effectively equivalent, then choose a smooth one.
    # This avoids unnecessary roll jumps when several angles have nearly same score.
    keep_margin_deg = 0.25
    near_opt = [c for c in roll_candidates if c[1] >= max_score - keep_margin_deg]

    if prev_roll_deg is None:
        best_roll_deg, _, best_y, best_z = max(near_opt, key=lambda c: c[1])
    else:
        def wrap_delta_deg(a, b):
            # Circular difference in [-180, 180] mapped to magnitude.
            return abs(((a - b + 180.0) % 360.0) - 180.0)

        best_roll_deg, _, best_y, best_z = min(
            near_opt,
            key=lambda c: wrap_delta_deg(c[0], prev_roll_deg)
        )

    return best_roll_deg, best_y, best_z, max_score

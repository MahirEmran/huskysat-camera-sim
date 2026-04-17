import numpy as np
from Basilisk.utilities import macros, RigidBodyKinematics as rbk
from Basilisk.architecture import sysModel, messaging

from guidance_math import approx_sun_hat_from_epoch, compute_compromise_x, solve_roll_for_lost_clearance


class ActiveGuidance(sysModel.SysModel):
    def __init__(self, mode, epoch_iso_utc, lost_excl_half_deg, status_period_sec, pos_found_b):
        super().__init__()
        self.mode = mode
        self.attRefOutMsg = messaging.AttRefMsg()
        self.scStateInMsg = messaging.SCStatesMsgReader()
        self.sunStateInMsg = messaging.SpicePlanetStateMsgReader()
        self.prev_roll_deg = None
        self.state = None
        self.pos_found_b = np.array(pos_found_b, dtype=float)

        # Bootstrap value used until the SPICE message is available.
        self.default_sun_hat = approx_sun_hat_from_epoch(epoch_iso_utc)

        # Earth apparent half-angle from a 400 km altitude shell.
        # This is used for the limb geometry and eclipse-style checks.
        self.rho_rad = np.arcsin(6371.0 / (6371.0 + 400.0))
        self.earth_half_angle_deg = np.degrees(self.rho_rad)

        # Hysteresis bands prevent CHARGING/EXPERIMENT chatter near boundaries.
        self.charge_enter_clear_deg = lost_excl_half_deg + 2.0
        self.charge_exit_clear_deg = max(lost_excl_half_deg - 1.0, 0.0)
        self.charge_enter_sun_vis_deg = self.earth_half_angle_deg + 2.0
        self.charge_exit_sun_vis_deg = max(self.earth_half_angle_deg - 1.0, 0.0)
        self.status_period_nanos = macros.sec2nano(status_period_sec)
        self.last_status_print_nanos = None

    def Reset(self, CurrentSimNanos):
        pass

    def UpdateState(self, CurrentSimNanos):
        scState = self.scStateInMsg()
        r_N = np.array(scState.r_BN_N)
        r_mag = np.linalg.norm(r_N)

        # Early startup can deliver effectively-empty state values; hold identity ref then.
        if r_mag < 1.0:
            refMsg = messaging.AttRefMsgPayload()
            refMsg.sigma_RN = [0.0, 0.0, 0.0]
            refMsg.omega_RN_N = [0.0, 0.0, 0.0]
            refMsg.domega_RN_N = [0.0, 0.0, 0.0]
            self.attRefOutMsg.write(refMsg, CurrentSimNanos)
            return

        # Use SPICE Sun ephemeris once available.
        # SPICE gives absolute positions, so subtract the spacecraft position.
        sun_hat_sc = self.default_sun_hat
        sun_abs_N = None
        if self.sunStateInMsg.isLinked() and self.sunStateInMsg.isWritten():
            sun_state = self.sunStateInMsg()
            sun_abs_N = np.array(sun_state.PositionVector)
            sun_rel_N = sun_abs_N - r_N
            sun_rel_mag = np.linalg.norm(sun_rel_N)
            if sun_rel_mag > 1.0:
                sun_hat_sc = sun_rel_N / sun_rel_mag

        earth_hat_sc = -r_N / r_mag  # Nadir direction in inertial coordinates.

        # FOUND is physically offset from COM, so use camera location (not COM) for constraints.
        found_pos_N = r_N.copy()
        if hasattr(scState, "sigma_BN"):
            try:
                sigma_BN_now = np.array(scState.sigma_BN)
                if sigma_BN_now.size == 3:
                    c_bn = rbk.MRP2C(sigma_BN_now)
                    c_nb = c_bn.T
                    found_pos_N = r_N + c_nb.dot(self.pos_found_b)
            except Exception:
                found_pos_N = r_N.copy()

        found_r_mag = np.linalg.norm(found_pos_N)
        # Using FOUND position (not COM) slightly changes Earth/Sun direction vectors,
        # which matters when constraints are close to cone boundaries.
        earth_hat_found = earth_hat_sc if found_r_mag < 1.0 else (-found_pos_N / found_r_mag)

        sun_hat_found = sun_hat_sc
        if sun_abs_N is not None:
            sun_rel_found_N = sun_abs_N - found_pos_N
            sun_rel_found_mag = np.linalg.norm(sun_rel_found_N)
            if sun_rel_found_mag > 1.0:
                sun_hat_found = sun_rel_found_N / sun_rel_found_mag

        # Roll-only wants -X at Sun so +X (FOUND) stays anti-sun.
        roll_only_x = -sun_hat_found
        roll_only_roll, roll_only_y, roll_only_z, roll_only_score = solve_roll_for_lost_clearance(
            roll_only_x, earth_hat_sc, sun_hat_sc, self.prev_roll_deg
        )

        # Angle between Sun line-of-sight and Earth center line-of-sight.
        # If this is smaller than Earth apparent half-angle, Sun is geometrically eclipsed.
        ang_sun_earth_deg = np.degrees(np.arccos(np.clip(np.dot(sun_hat_found, earth_hat_found), -1.0, 1.0)))

        if self.mode == "ROLL_ONLY":
            x_B = roll_only_x
            best_roll_deg, best_y, best_z = roll_only_roll, roll_only_y, roll_only_z
            selected_state = "CHARGING"
        elif self.mode in ("EXPERIMENT", "COMPROMISE"):
            # Keep FOUND near Earth limb but bias away from Sun.
            x_B = compute_compromise_x(earth_hat_found, sun_hat_found, self.rho_rad)
            best_roll_deg, best_y, best_z, _ = solve_roll_for_lost_clearance(
                x_B, earth_hat_sc, sun_hat_sc, self.prev_roll_deg
            )
            selected_state = "EXPERIMENT"
        else:
            # HYBRID state machine.
            # Enter thresholds are stricter than exit thresholds (intentional hysteresis).
            if self.state == "CHARGING":
                can_charge = roll_only_score >= self.charge_exit_clear_deg
                sun_visible = ang_sun_earth_deg >= self.charge_exit_sun_vis_deg
            else:
                can_charge = roll_only_score >= self.charge_enter_clear_deg
                sun_visible = ang_sun_earth_deg >= self.charge_enter_sun_vis_deg

            can_charge = can_charge and sun_visible

            if can_charge:
                x_B = roll_only_x
                best_roll_deg, best_y, best_z = roll_only_roll, roll_only_y, roll_only_z
                selected_state = "CHARGING"
            else:
                # Fallback to experiment mode when either LOST clearance or Sun visibility is insufficient.
                x_B = compute_compromise_x(earth_hat_found, sun_hat_found, self.rho_rad)
                best_roll_deg, best_y, best_z, _ = solve_roll_for_lost_clearance(
                    x_B, earth_hat_sc, sun_hat_sc, self.prev_roll_deg
                )
                selected_state = "EXPERIMENT"

        if selected_state != self.state:
            t_sec = CurrentSimNanos * 1.0e-9
            panel_cmd = "OPEN" if selected_state == "CHARGING" else "STOW"
            print(
                f"[ADCS] t={t_sec:8.1f}s -> {selected_state} "
                f"(roll-only LOST clearance={roll_only_score:5.1f} deg, panel cmd={panel_cmd})"
            )
            self.state = selected_state

        if self.mode == "HYBRID":
            should_print_status = (
                self.last_status_print_nanos is None
                or (CurrentSimNanos - self.last_status_print_nanos) >= self.status_period_nanos
            )
            if should_print_status:
                t_sec = CurrentSimNanos * 1.0e-9
                panel_cmd = "OPEN" if selected_state == "CHARGING" else "STOW"
                print(
                    f"[HYBRID] t={t_sec:8.1f}s active={selected_state} "
                    f"panel={panel_cmd} roll-only-clearance={roll_only_score:5.1f} deg "
                    f"sun-earth-angle={ang_sun_earth_deg:5.1f} deg"
                )
                self.last_status_print_nanos = CurrentSimNanos

        self.prev_roll_deg = best_roll_deg

        # Assemble DCM rows [x_B, y_B, z_B] and convert to MRP.
        # Here we define R relative to N by its body axes expressed in inertial coordinates.
        dcm_RN = np.array([x_B, best_y, best_z])
        sigma_RN = rbk.C2MRP(dcm_RN)

        # Rare numerical edge case guard.
        if np.any(np.isnan(sigma_RN)):
            sigma_RN = np.array([0.0, 0.0, 0.0])

        refMsg = messaging.AttRefMsgPayload()
        refMsg.sigma_RN = sigma_RN.tolist()
        refMsg.omega_RN_N = [0.0, 0.0, 0.0]
        refMsg.domega_RN_N = [0.0, 0.0, 0.0]
        self.attRefOutMsg.write(refMsg, CurrentSimNanos)

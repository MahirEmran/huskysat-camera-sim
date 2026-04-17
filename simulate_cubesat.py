import os
import argparse
import subprocess
import sys
import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, orbitalMotion, RigidBodyKinematics as rbk
from Basilisk.simulation import spacecraft, extForceTorque, simpleNav
from Basilisk.fswAlgorithms import attTrackingError, mrpPD
from Basilisk.architecture import messaging

from guidance_math import approx_sun_hat_from_epoch, compute_compromise_x, solve_roll_for_lost_clearance
from active_guidance import ActiveGuidance
from visual_model import build_satellite_obj, apply_visual_model
from uptime_metrics import compute_camera_uptime_flags
from vizard_scene import enable_vizard, add_vizard_scene_overlays


DEFAULT_ADCS_MODE = "HYBRID"
DEFAULT_SIM_HOURS = 24.0
DEFAULT_BODY_X_M = 0.30
DEFAULT_BODY_YZ_M = 0.10
DEFAULT_LOST_FOV_DEG = 15.0
DEFAULT_FOUND_FOV_DEG = 60.0
DEFAULT_EXCLUSION_BUFFER_DEG = 10.0
DEFAULT_STATUS_PERIOD_SEC = 60.0


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run CubeSat camera simulation with selectable ADCS mode.")
    parser.add_argument(
        "--mode",
        choices=["ROLL_ONLY", "EXPERIMENT", "COMPROMISE", "HYBRID", "BOTH"],
        default=DEFAULT_ADCS_MODE,
        type=str.upper,
        help="ADCS mode to run (default: HYBRID). COMPROMISE is accepted as a legacy alias for EXPERIMENT.",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=DEFAULT_SIM_HOURS,
        help="Simulation duration in hours (default: 1).",
    )
    parser.add_argument(
        "--body-x",
        type=float,
        default=DEFAULT_BODY_X_M,
        help="CubeSat long body dimension in meters (legacy name); applied along +Z so +/-Z are square faces (default: 0.30).",
    )
    parser.add_argument(
        "--body-yz",
        type=float,
        default=DEFAULT_BODY_YZ_M,
        help="CubeSat square face side length for X/Y in meters (default: 0.10).",
    )
    parser.add_argument(
        "--lost-fov",
        type=float,
        default=DEFAULT_LOST_FOV_DEG,
        help="LOST camera full FOV in degrees (default: 15).",
    )
    parser.add_argument(
        "--found-fov",
        type=float,
        default=DEFAULT_FOUND_FOV_DEG,
        help="FOUND camera full FOV in degrees (default: 60).",
    )
    parser.add_argument(
        "--exclusion-buffer",
        type=float,
        default=DEFAULT_EXCLUSION_BUFFER_DEG,
        help="Exclusion cone buffer in degrees added to half-FOV (default: 10).",
    )
    parser.add_argument(
        "--status-period",
        type=float,
        default=DEFAULT_STATUS_PERIOD_SEC,
        help="Status print period in seconds for HYBRID mode (default: 60).",
    )
    args = parser.parse_args()
    if args.hours <= 0.0:
        parser.error("--hours must be greater than 0")
    if args.body_x <= 0.0 or args.body_yz <= 0.0:
        parser.error("--body-x and --body-yz must be greater than 0")
    if args.lost_fov <= 0.0 or args.found_fov <= 0.0:
        parser.error("--lost-fov and --found-fov must be greater than 0")
    if args.exclusion_buffer < 0.0:
        parser.error("--exclusion-buffer must be >= 0")
    if args.status_period <= 0.0:
        parser.error("--status-period must be greater than 0")
    return args


CLI_ARGS = parse_cli_args()


def run_both_modes(args):
    script_path = os.path.abspath(__file__)
    shared = [
        "--hours", str(args.hours),
        "--body-x", str(args.body_x),
        "--body-yz", str(args.body_yz),
        "--lost-fov", str(args.lost_fov),
        "--found-fov", str(args.found_fov),
        "--exclusion-buffer", str(args.exclusion_buffer),
        "--status-period", str(args.status_period),
    ]
    commands = [
        [sys.executable, script_path, "--mode", "ROLL_ONLY", *shared],
        [sys.executable, script_path, "--mode", "EXPERIMENT", *shared],
    ]

    print("Mode: BOTH (launching ROLL_ONLY and EXPERIMENT subprocesses)")
    processes = [subprocess.Popen(cmd) for cmd in commands]
    return_codes = [proc.wait() for proc in processes]

    if any(code != 0 for code in return_codes):
        raise SystemExit(max(return_codes))
    raise SystemExit(0)

# mode for ADCS
# - "ROLL_ONLY": keep -X on Sun (so +X FOUND faces away); roll to maximize LOST (+Z) sky clearance
# - "EXPERIMENT": point FOUND (+X) toward Earth limb and try to roll for LOST (+Z) uptime
# - "HYBRID": switch between CHARGING (ROLL_ONLY) and EXPERIMENT automatically
# - "BOTH": spawn two subprocesses and run fixed ROLL_ONLY and fixed EXPERIMENT for comparison
ADCS_MODE = CLI_ARGS.mode

# Backward-compatible rename: COMPROMISE -> EXPERIMENT.
if ADCS_MODE == "COMPROMISE":
    ADCS_MODE = "EXPERIMENT"

if ADCS_MODE == "BOTH":
    run_both_modes(CLI_ARGS)

# simulation start time - used for sun estimate + SPICE
SIM_EPOCH_UTC = "2026-04-01T12:00:00.000Z"

# body frame layout:
#   +X = long rectangular side  → FOUND camera face
#   +Z = square endcap face     → LOST camera face
#   -Z = opposite square endcap → antenna
# NOTE: keep CLI flag names for compatibility, but map dimensions to match the face layout above.
BODY_LONG_M = CLI_ARGS.body_x
BODY_SIDE_M = CLI_ARGS.body_yz

BODY_SIZE_XY_M = BODY_SIDE_M
BODY_SIZE_Z_M = BODY_LONG_M

BODY_SIZE_X_M = BODY_SIZE_XY_M
BODY_SIZE_Y_M = BODY_SIZE_XY_M
FOUND_Z_OFFSET_FRAC = 0.35

# sensor boresight vectors in body frame
VEC_LOST_B  = [0, 0,  1]   # LOST points out +Z (square face)
VEC_FOUND_B = [1, 0,  0]   # FOUND points out +X (long face)
VEC_ANT_B   = [0, 0, -1]   # antenna points out -Z (opposite square face)

# sensor positions at center of each face
POS_LOST_B  = [0.0, 0.0,  0.5 * BODY_SIZE_Z_M]
POS_FOUND_B = [0.5 * BODY_SIZE_X_M, 0.0, FOUND_Z_OFFSET_FRAC * BODY_SIZE_Z_M]
POS_ANT_B   = [0.0, 0.0, -0.5 * BODY_SIZE_Z_M]

# camera FOVs (full angle) and derived half-angles
LOST_FOV_DEG  = CLI_ARGS.lost_fov
FOUND_FOV_DEG = CLI_ARGS.found_fov

LOST_HALF_DEG  = LOST_FOV_DEG  / 2.0
FOUND_HALF_DEG = FOUND_FOV_DEG / 2.0
EXCLUSION_BUFFER_DEG = CLI_ARGS.exclusion_buffer

# exclusion cones are a bit wider than the strict FOV as a safety margin
LOST_EXCL_HALF_DEG  = LOST_HALF_DEG  + EXCLUSION_BUFFER_DEG
FOUND_EXCL_HALF_DEG = FOUND_HALF_DEG + EXCLUSION_BUFFER_DEG

SIM_HOURS = CLI_ARGS.hours

# pre-compute initial attitude so the controller starts near the target
# and never has to traverse a large MRP arc at t=0 (that would spike the torque)
sun_hat = approx_sun_hat_from_epoch(SIM_EPOCH_UTC)

mu_init = 3.986004415e14
oe_init = orbitalMotion.ClassicElements()
oe_init.a = (6371.0 + 400.0) * 1000.0  # 400 km circular
oe_init.e = 0.0001; oe_init.i = 51.6 * macros.D2R  # ISS-like inclination
oe_init.Omega = 0.0; oe_init.omega = 0.0; oe_init.f = 90.0 * macros.D2R
rN_init, _     = orbitalMotion.elem2rv(mu_init, oe_init)
earth_hat_init = -np.array(rN_init) / np.linalg.norm(rN_init)

if ADCS_MODE == "ROLL_ONLY":
    x_B_init = -sun_hat
    init_state = "CHARGING"
elif ADCS_MODE == "EXPERIMENT":
    rho_rad = np.arcsin(6371.0 / (6371.0 + 400.0))
    x_B_init = compute_compromise_x(earth_hat_init, sun_hat, rho_rad)
    init_state = "EXPERIMENT"
else:
    # HYBRID and BOTH startup: choose CHARGING only when roll-only has healthy LOST clearance.
    charge_enter_clear_deg = LOST_EXCL_HALF_DEG + 2.0
    earth_half_angle_deg = np.degrees(np.arcsin(6371.0 / (6371.0 + 400.0)))
    charge_enter_sun_vis_deg = earth_half_angle_deg + 2.0
    _, _, _, charge_score_init = solve_roll_for_lost_clearance(-sun_hat, earth_hat_init, sun_hat)
    sun_earth_ang_init = np.degrees(np.arccos(np.clip(np.dot(sun_hat, earth_hat_init), -1.0, 1.0)))
    if charge_score_init >= charge_enter_clear_deg and sun_earth_ang_init >= charge_enter_sun_vis_deg:
        x_B_init = -sun_hat
        init_state = "CHARGING"
    else:
        rho_rad = np.arcsin(6371.0 / (6371.0 + 400.0))
        x_B_init = compute_compromise_x(earth_hat_init, sun_hat, rho_rad)
        init_state = "EXPERIMENT"

_, y_B_init, z_B_init, _ = solve_roll_for_lost_clearance(x_B_init, earth_hat_init, sun_hat)
sigma_BN_init = rbk.C2MRP(np.array([x_B_init, y_B_init, z_B_init]))
print(f"Initial sigma_BN (near target): {sigma_BN_init.tolist()}")
print(f"Initial ADCS state: {init_state}")


scSim       = SimulationBaseClass.SimBaseClass()
simTaskName = "dynamicsTask"
dynProcess  = scSim.CreateNewProcess("dynamicsProcess")

# 0.5 s timestep: ω_n*dt = 0.707*0.5 = 0.35 << π  (stable)
dynProcess.addTask(scSim.CreateNewTask(simTaskName, macros.sec2nano(0.5)))

scObject          = spacecraft.Spacecraft()
scObject.ModelTag = "cubesat"
scObject.hub.mHub = 12.0
scObject.hub.IHubPntBc_B = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]  # kg*m^2, roughly 3U-ish
scSim.AddModelToTask(simTaskName, scObject)

gravFactory = simIncludeGravBody.gravBodyFactory()
earth = gravFactory.createEarth(); earth.isCentralBody = True
sun   = gravFactory.createSun()
moon  = gravFactory.createMoon()
gravFactory.createSpiceInterface(time=SIM_EPOCH_UTC, epochInMsg=True)
gravFactory.spiceObject.zeroBase = "earth"  # all positions relative to Earth center
scSim.AddModelToTask(simTaskName, gravFactory.spiceObject)
scObject.gravField.gravBodies = spacecraft.GravBodyVector(list(gravFactory.gravBodies.values()))

mu   = 3.986004415e14
oe   = orbitalMotion.ClassicElements()
oe.a = (6371.0 + 400.0) * 1000.0
oe.e = 0.0001; oe.i = 51.6 * macros.D2R
oe.Omega = 0.0; oe.omega = 0.0; oe.f = 90.0 * macros.D2R
rN, vN = orbitalMotion.elem2rv(mu, oe)
scObject.hub.r_CN_NInit     = rN
scObject.hub.v_CN_NInit     = vN
scObject.hub.sigma_BNInit   = sigma_BN_init.tolist()  # near target, not identity
scObject.hub.omega_BN_BInit = [0.0, 0.0, 0.0]

sNav          = simpleNav.SimpleNav()
sNav.ModelTag = "SimpleNav"
scSim.AddModelToTask(simTaskName, sNav)
sNav.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

guidance = ActiveGuidance(
    mode=ADCS_MODE,
    epoch_iso_utc=SIM_EPOCH_UTC,
    lost_excl_half_deg=LOST_EXCL_HALF_DEG,
    status_period_sec=CLI_ARGS.status_period,
    pos_found_b=POS_FOUND_B,
)
guidance.ModelTag = "activeGuidance"
guidance.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

# wire up the SPICE Sun message so guidance gets a real-time Sun direction
sun_index = 1
moon_index = 2
if hasattr(gravFactory, "spicePlanetNames"):
    planet_names = [name.lower() for name in gravFactory.spicePlanetNames]
    if "sun" in planet_names:
        sun_index = planet_names.index("sun")
    if "moon" in planet_names:
        moon_index = planet_names.index("moon")
guidance.sunStateInMsg.subscribeTo(gravFactory.spiceObject.planetStateOutMsgs[sun_index])

scSim.AddModelToTask(simTaskName, guidance)

attErr          = attTrackingError.attTrackingError()
attErr.ModelTag = "attError"
scSim.AddModelToTask(simTaskName, attErr)
attErr.attNavInMsg.subscribeTo(sNav.attOutMsg)
attErr.attRefInMsg.subscribeTo(guidance.attRefOutMsg)

# MRP PD gains: K=0.05, P=0.3 give ω_n = 0.707 rad/s, ζ ≈ 2.1 (overdamped), stable at 0.5 s
mrpControl          = mrpPD.mrpPD()
mrpControl.ModelTag = "mrpPD"
mrpControl.K        = 0.05
mrpControl.P        = 0.3
scSim.AddModelToTask(simTaskName, mrpControl)
mrpControl.guidInMsg.subscribeTo(attErr.attGuidOutMsg)

vehicleConfig = messaging.VehicleConfigMsgPayload()
vehicleConfig.ISCPntB_B = [0.1, 0.0, 0.0,
                            0.0, 0.1, 0.0,
                            0.0, 0.0, 0.1]
vehConfigMsg = messaging.VehicleConfigMsg().write(vehicleConfig)
mrpControl.vehConfigInMsg.subscribeTo(vehConfigMsg)

extFT          = extForceTorque.ExtForceTorque()
extFT.ModelTag = "extFT"
scSim.AddModelToTask(simTaskName, extFT)
scObject.addDynamicEffector(extFT)
extFT.cmdTorqueInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)

save_path = os.path.join(os.getcwd(), f'simulation_{ADCS_MODE}.bin')
if os.path.exists(save_path):
    os.remove(save_path)  # clear old run so Vizard doesn't load stale data

# Vizard bootstrap is handled in one helper so this file can stay focused on sim logic.
viz = enable_vizard(scSim, simTaskName, scObject, save_path)

# Pre-build both visual variants once (OPEN/CLOSED) and hot-swap them during runtime.
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)
obj_path_open = os.path.join(models_dir, f"cubesat_{ADCS_MODE}_open.obj")
obj_path_closed = os.path.join(models_dir, f"cubesat_{ADCS_MODE}_closed.obj")
build_satellite_obj(
    obj_path_open,
    panels_open=True,
    body_size_x_m=BODY_SIZE_X_M,
    body_size_y_m=BODY_SIZE_Y_M,
    body_size_z_m=BODY_SIZE_Z_M,
)
build_satellite_obj(
    obj_path_closed,
    panels_open=False,
    body_size_x_m=BODY_SIZE_X_M,
    body_size_y_m=BODY_SIZE_Y_M,
    body_size_z_m=BODY_SIZE_Z_M,
)

initial_visual_state = "OPEN" if init_state == "CHARGING" else "CLOSED"
initial_visual_path = obj_path_open if initial_visual_state == "OPEN" else obj_path_closed
apply_visual_model(viz, scObject.ModelTag, initial_visual_path)
print(f"Visual model: panels={initial_visual_state} (based on initial ADCS state={init_state})")

# Add all cones/lines/cameras used to interpret LOST and FOUND geometry.
add_vizard_scene_overlays(
    viz,
    spacecraft_tag=scObject.ModelTag,
    vec_lost_b=VEC_LOST_B,
    vec_found_b=VEC_FOUND_B,
    pos_lost_b=POS_LOST_B,
    pos_found_b=POS_FOUND_B,
    lost_fov_deg=LOST_FOV_DEG,
    found_fov_deg=FOUND_FOV_DEG,
    lost_half_deg=LOST_HALF_DEG,
    found_half_deg=FOUND_HALF_DEG,
    lost_excl_half_deg=LOST_EXCL_HALF_DEG,
    found_excl_half_deg=FOUND_EXCL_HALF_DEG,
)

print(f"Mode: {ADCS_MODE}")
scSim.InitializeSimulation()
stop_time_nanos = macros.hour2nano(SIM_HOURS)
visual_update_step_nanos = macros.sec2nano(5.0)
next_stop_nanos = 0
current_visual_state = None
# current_visual_state = initial_visual_state

sampled_time_nanos = 0
lost_uptime_nanos = 0
found_uptime_nanos = 0

sampled_time_charging_nanos = 0
sampled_time_experiment_nanos = 0
lost_uptime_charging_nanos = 0
lost_uptime_experiment_nanos = 0
found_uptime_charging_nanos = 0
found_uptime_experiment_nanos = 0

uptime_warning_printed = False

# Step in chunks instead of one long run: this gives us model swaps and uptime sampling hooks.
while next_stop_nanos < stop_time_nanos:
    previous_stop_nanos = next_stop_nanos
    next_stop_nanos = min(next_stop_nanos + visual_update_step_nanos, stop_time_nanos)
    scSim.ConfigureStopTime(next_stop_nanos)
    scSim.ExecuteSimulation()

    dt_nanos = next_stop_nanos - previous_stop_nanos
    if dt_nanos > 0:
        sampled_time_nanos += dt_nanos
        try:
            sc_state_now = scObject.scStateOutMsg.read()
            sun_state_now = gravFactory.spiceObject.planetStateOutMsgs[sun_index].read()
            moon_state_now = gravFactory.spiceObject.planetStateOutMsgs[moon_index].read()
            lost_ok, found_ok = compute_camera_uptime_flags(
                sc_state_now,
                sun_state_now,
                moon_state_now,
                vec_lost_b=VEC_LOST_B,
                vec_found_b=VEC_FOUND_B,
                pos_lost_b=POS_LOST_B,
                pos_found_b=POS_FOUND_B,
                # Uptime is tied to INNER red cone validity, not outer orange buffers.
                lost_inner_keepout_half_deg=LOST_HALF_DEG,
                found_earth_keepin_half_deg=FOUND_HALF_DEG,
                found_sun_keepout_half_deg=FOUND_HALF_DEG,
            )
            if lost_ok:
                lost_uptime_nanos += dt_nanos
            if found_ok:
                found_uptime_nanos += dt_nanos

            # Track per-state uptime so CHARGING and EXPERIMENT performance are visible separately.
            sample_state = guidance.state
            if sample_state not in ("CHARGING", "EXPERIMENT"):
                if ADCS_MODE == "ROLL_ONLY":
                    sample_state = "CHARGING"
                elif ADCS_MODE == "EXPERIMENT":
                    sample_state = "EXPERIMENT"

            if sample_state == "CHARGING":
                sampled_time_charging_nanos += dt_nanos
                if lost_ok:
                    lost_uptime_charging_nanos += dt_nanos
                if found_ok:
                    found_uptime_charging_nanos += dt_nanos
            elif sample_state == "EXPERIMENT":
                sampled_time_experiment_nanos += dt_nanos
                if lost_ok:
                    lost_uptime_experiment_nanos += dt_nanos
                if found_ok:
                    found_uptime_experiment_nanos += dt_nanos
        except Exception as exc:
            if not uptime_warning_printed:
                print(f"[UPTIME] sample evaluation warning: {exc}")
                uptime_warning_printed = True

    # Visual state mirrors guidance mode: OPEN panels while charging, hidden panels otherwise.
    if guidance.state == "CHARGING":
        target_visual_state = "OPEN"
    elif guidance.state == "EXPERIMENT":
        target_visual_state = "CLOSED"
    else:
        target_visual_state = current_visual_state

    if target_visual_state != current_visual_state:
        target_model = obj_path_open if target_visual_state == "OPEN" else obj_path_closed
        apply_visual_model(viz, scObject.ModelTag, target_model)
        print(f"[VIZ] t={next_stop_nanos * 1.0e-9:8.1f}s switched visual model -> {target_visual_state}")
        current_visual_state = target_visual_state

if sampled_time_nanos > 0:
    lost_uptime_pct = 100.0 * lost_uptime_nanos / sampled_time_nanos
    found_uptime_pct = 100.0 * found_uptime_nanos / sampled_time_nanos
else:
    lost_uptime_pct = 0.0
    found_uptime_pct = 0.0

if sampled_time_charging_nanos > 0:
    lost_uptime_charging_pct = 100.0 * lost_uptime_charging_nanos / sampled_time_charging_nanos
    found_uptime_charging_pct = 100.0 * found_uptime_charging_nanos / sampled_time_charging_nanos
else:
    lost_uptime_charging_pct = None
    found_uptime_charging_pct = None

if sampled_time_experiment_nanos > 0:
    lost_uptime_experiment_pct = 100.0 * lost_uptime_experiment_nanos / sampled_time_experiment_nanos
    found_uptime_experiment_pct = 100.0 * found_uptime_experiment_nanos / sampled_time_experiment_nanos
else:
    lost_uptime_experiment_pct = None
    found_uptime_experiment_pct = None

print(f"[UPTIME] LOST clear uptime (total):    {lost_uptime_pct:6.2f}%")
print(f"[UPTIME] FOUND valid uptime (total):   {found_uptime_pct:6.2f}%")
if lost_uptime_charging_pct is None:
    print("[UPTIME] LOST clear uptime (CHARGING): N/A (state not visited)")
else:
    print(f"[UPTIME] LOST clear uptime (CHARGING): {lost_uptime_charging_pct:6.2f}%")
if found_uptime_charging_pct is None:
    print("[UPTIME] FOUND valid uptime (CHARGING): N/A (state not visited)")
else:
    print(f"[UPTIME] FOUND valid uptime (CHARGING): {found_uptime_charging_pct:6.2f}%")
if lost_uptime_experiment_pct is None:
    print("[UPTIME] LOST clear uptime (EXPERIMENT): N/A (state not visited)")
else:
    print(f"[UPTIME] LOST clear uptime (EXPERIMENT): {lost_uptime_experiment_pct:6.2f}%")
if found_uptime_experiment_pct is None:
    print("[UPTIME] FOUND valid uptime (EXPERIMENT): N/A (state not visited)")
else:
    print(f"[UPTIME] FOUND valid uptime (EXPERIMENT): {found_uptime_experiment_pct:6.2f}%")

print(f"Done! Load {save_path} in Vizard.")

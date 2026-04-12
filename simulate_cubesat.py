import os
import argparse
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, orbitalMotion, vizSupport, RigidBodyKinematics as rbk
from Basilisk.simulation import spacecraft, extForceTorque, simpleNav
from Basilisk.fswAlgorithms import attTrackingError, mrpPD
from Basilisk.architecture import sysModel, messaging


DEFAULT_ADCS_MODE = "BOTH"
DEFAULT_SIM_HOURS = 12.0
DEFAULT_BODY_X_M = 0.30
DEFAULT_BODY_YZ_M = 0.10
DEFAULT_LOST_FOV_DEG = 15.0
DEFAULT_FOUND_FOV_DEG = 60.0
DEFAULT_EXCLUSION_BUFFER_DEG = 10.0


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run CubeSat camera simulation with selectable ADCS mode.")
    parser.add_argument(
        "--mode",
        choices=["ROLL_ONLY", "COMPROMISE", "BOTH"],
        default=DEFAULT_ADCS_MODE,
        type=str.upper,
        help="ADCS mode to run (default: BOTH).",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=DEFAULT_SIM_HOURS,
        help="Simulation duration in hours (default: 12).",
    )
    parser.add_argument(
        "--body-x",
        type=float,
        default=DEFAULT_BODY_X_M,
        help="CubeSat body length along +X in meters (default: 0.30).",
    )
    parser.add_argument(
        "--body-yz",
        type=float,
        default=DEFAULT_BODY_YZ_M,
        help="CubeSat square base side length for Y/Z in meters (default: 0.10).",
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
    args = parser.parse_args()
    if args.hours <= 0.0:
        parser.error("--hours must be greater than 0")
    if args.body_x <= 0.0 or args.body_yz <= 0.0:
        parser.error("--body-x and --body-yz must be greater than 0")
    if args.lost_fov <= 0.0 or args.found_fov <= 0.0:
        parser.error("--lost-fov and --found-fov must be greater than 0")
    if args.exclusion_buffer < 0.0:
        parser.error("--exclusion-buffer must be >= 0")
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
    ]
    commands = [
        [sys.executable, script_path, "--mode", "ROLL_ONLY", *shared],
        [sys.executable, script_path, "--mode", "COMPROMISE", *shared],
    ]

    print("Mode: BOTH (launching ROLL_ONLY and COMPROMISE subprocesses)")
    processes = [subprocess.Popen(cmd) for cmd in commands]
    return_codes = [proc.wait() for proc in processes]

    if any(code != 0 for code in return_codes):
        raise SystemExit(max(return_codes))
    raise SystemExit(0)

# mode for ADCS
# - "ROLL_ONLY": keep -X on Sun, prioritizing LOST uptime
# - "COMPROMISE": point FOUND (+X) toward Earth limb and try to roll for LOST uptime
# - "BOTH": spawn two subprocesses and run both modes
ADCS_MODE = CLI_ARGS.mode

if ADCS_MODE == "BOTH":
    run_both_modes(CLI_ARGS)

# simulation start time - used for sun estimate + SPICE
SIM_EPOCH_UTC = "2026-04-01T12:00:00.000Z"

# dimensions of cubesat (Y=Z)
BODY_LENGTH_X_M = CLI_ARGS.body_x
BODY_BASE_YZ_M = CLI_ARGS.body_yz

# sensor vectors in body frame
VEC_LOST_B = [0, 0, -1]
VEC_FOUND_B = [1, 0, 0]

# Place sensors at the center of each corresponding face.
POS_LOST_B = [0.0, 0.0, -0.5 * BODY_BASE_YZ_M]
POS_FOUND_B = [0.5 * BODY_LENGTH_X_M, 0.0, 0.0]

# Camera FOVs (degrees) and cone half-angles (degrees).
LOST_FOV_DEG = CLI_ARGS.lost_fov
FOUND_FOV_DEG = CLI_ARGS.found_fov

LOST_HALF_DEG = LOST_FOV_DEG / 2.0
FOUND_HALF_DEG = FOUND_FOV_DEG / 2.0
EXCLUSION_BUFFER_DEG = CLI_ARGS.exclusion_buffer

LOST_EXCL_HALF_DEG = LOST_HALF_DEG + EXCLUSION_BUFFER_DEG
FOUND_EXCL_HALF_DEG = FOUND_HALF_DEG + EXCLUSION_BUFFER_DEG

# number of hours to run sim for
SIM_HOURS = CLI_ARGS.hours



def approx_sun_hat_from_epoch(epoch_iso_utc):
    """Approximate inertial Earth->Sun unit vector from epoch (UTC ISO-8601)."""
    # Parse "...Z" as UTC and keep sub-second precision when present.
    dt = datetime.fromisoformat(epoch_iso_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    # Days since J2000 epoch (2000-01-01T12:00:00Z).
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    n_days = (dt - j2000).total_seconds() / 86400.0
    # Low-order solar ephemeris approximation in degrees.
    mean_long = np.radians((280.460 + 0.9856474 * n_days) % 360.0)
    mean_anom = np.radians((357.528 + 0.9856003 * n_days) % 360.0)
    # Apparent ecliptic longitude and obliquity.
    lam = mean_long + np.radians(1.915) * np.sin(mean_anom) + np.radians(0.020) * np.sin(2.0 * mean_anom)
    eps = np.radians(23.439 - 0.0000004 * n_days)
    # Earth->Sun direction in inertial frame (same convention used elsewhere in this script).
    sun_hat = np.array([
        np.cos(lam),
        np.cos(eps) * np.sin(lam),
        np.sin(eps) * np.sin(lam),
    ])
    return sun_hat / np.linalg.norm(sun_hat)


class ActiveGuidance(sysModel.SysModel):
    def __init__(self, mode="ROLL_ONLY", epoch_iso_utc=SIM_EPOCH_UTC):
        super().__init__()
        self.mode = mode
        self.attRefOutMsg = messaging.AttRefMsg()
        self.scStateInMsg = messaging.SCStatesMsgReader()
        self.sunStateInMsg = messaging.SpicePlanetStateMsgReader()
        self.prev_roll_deg = None
        # need some initial sun thing, it'll be overridden by actual simulation later
        self.default_sun_hat = approx_sun_hat_from_epoch(epoch_iso_utc)
        # half-angle of Earth's disk as seen from 400 km: arcsin(R_e / (R_e + h))
        self.rho_rad = np.arcsin(6371.0 / (6371.0 + 400.0))

    def Reset(self, CurrentSimNanos):
        pass

    def UpdateState(self, CurrentSimNanos):
        scState = self.scStateInMsg()
        r_N     = np.array(scState.r_BN_N)
        r_mag   = np.linalg.norm(r_N)

        # guard against uninit message at t=0
        # if you don't have this you get bunch of bugs with NaN
        if r_mag < 1.0:
            refMsg             = messaging.AttRefMsgPayload()
            refMsg.sigma_RN    = [0.0, 0.0, 0.0]
            refMsg.omega_RN_N  = [0.0, 0.0, 0.0]
            refMsg.domega_RN_N = [0.0, 0.0, 0.0]
            self.attRefOutMsg.write(refMsg, CurrentSimNanos)
            return

        # use SPICE Sun ephemeris once it's live; fall back to the epoch approximation before first write
        sun_hat = self.default_sun_hat
        if self.sunStateInMsg.isLinked() and self.sunStateInMsg.isWritten():
            sun_state = self.sunStateInMsg()
            # SPICE gives absolute position, so subtract s/c position to get relative direction
            sun_rel_N = np.array(sun_state.PositionVector) - r_N
            sun_rel_mag = np.linalg.norm(sun_rel_N)
            if sun_rel_mag > 1.0:
                sun_hat = sun_rel_N / sun_rel_mag

        earth_hat = -r_N / r_mag  # nadir direction

        if self.mode == "ROLL_ONLY":
            # point -X directly at Sun so FOUND (+X) faces away; LOST gets best sky access
            x_B = -sun_hat

        elif self.mode == "COMPROMISE":
            # point +X toward the Earth limb at angle rho from nadir, on the anti-sun side
            # this keeps FOUND seeing Earth while pushing it away from the Sun
            u1       = earth_hat
            anti_sun = -sun_hat
            # project anti_sun onto the plane perpendicular to nadir to get the in-plane push direction
            u2       = anti_sun - np.dot(anti_sun, u1) * u1
            n2       = np.linalg.norm(u2)
            if n2 < 1e-6:  # sun is near nadir, pick any perpendicular
                u2 = np.array([1, 0, 0]) if abs(u1[0]) < 0.9 else np.array([0, 1, 0])
                u2 = u2 - np.dot(u2, u1) * u1
                n2 = np.linalg.norm(u2)
            u2 /= n2
            # x_B sits on the limb cone: cos(rho)*nadir + sin(rho)*away-from-sun
            x_B = np.cos(self.rho_rad) * u1 + np.sin(self.rho_rad) * u2

        x_B /= np.linalg.norm(x_B)

        # build an initial y/z frame around x_B so we have something to rotate
        # cross with earth_hat to get a stable z reference (avoids gimbal lock most of the time)
        z_temp = np.cross(x_B, earth_hat)
        if np.linalg.norm(z_temp) < 1e-6:  # x_B is nearly parallel to nadir, fall back
            z_temp = np.cross(x_B, np.array([0, 1, 0]))
            if np.linalg.norm(z_temp) < 1e-6:
                z_temp = np.cross(x_B, np.array([0, 0, 1]))
        z_temp /= np.linalg.norm(z_temp)
        y_temp  = np.cross(z_temp, x_B)
        y_temp /= np.linalg.norm(y_temp)

        # sweep roll angle in 2 deg steps and score each by how far LOST (-Z) is from Earth and Sun
        # the score is the min of the two angles so we maximize the worst-case clearance
        roll_candidates = []
        max_score = -1.0

        for roll_deg in range(0, 360, 2):
            roll     = np.radians(roll_deg)
            y_test   = np.cos(roll) * y_temp + np.sin(roll) * z_temp  # rotate y around x_B
            z_test   = np.cross(x_B, y_test)
            z_test  /= np.linalg.norm(z_test)
            lost_hat = -z_test  # LOST sensor points out -Z

            ang_earth = np.degrees(np.arccos(np.clip(np.dot(lost_hat, earth_hat), -1.0, 1.0)))
            ang_sun   = np.degrees(np.arccos(np.clip(np.dot(lost_hat, sun_hat),   -1.0, 1.0)))
            score     = min(ang_earth, ang_sun)  # limited by whichever body is closer

            roll_candidates.append((roll_deg, score, y_test, z_test))
            if score > max_score:
                max_score = score

        # keep roll continuous over time: among near-optimal roll choices,
        # prefer the one closest to the prior command to avoid 'jumps' in simulation
        keep_margin_deg = 0.25
        near_opt = [c for c in roll_candidates if c[1] >= max_score - keep_margin_deg]

        if self.prev_roll_deg is None:
            best_roll_deg, _, best_y, best_z = max(near_opt, key=lambda c: c[1])
        else:
            def wrap_delta_deg(a, b):
                # shortest angular distance on a circle
                return abs(((a - b + 180.0) % 360.0) - 180.0)

            best_roll_deg, _, best_y, best_z = min(
                near_opt,
                key=lambda c: wrap_delta_deg(c[0], self.prev_roll_deg)
            )

        self.prev_roll_deg = best_roll_deg

        # assemble DCM rows [x_B, y_B, z_B] and convert to MRP
        dcm_RN   = np.array([x_B, best_y, best_z])
        sigma_RN = rbk.C2MRP(dcm_RN)

        # if C2MRP produced NaN, hold identity
        if np.any(np.isnan(sigma_RN)):
            sigma_RN = np.array([0.0, 0.0, 0.0])

        refMsg             = messaging.AttRefMsgPayload()
        refMsg.sigma_RN    = sigma_RN.tolist()
        refMsg.omega_RN_N  = [0.0, 0.0, 0.0]
        refMsg.domega_RN_N = [0.0, 0.0, 0.0]
        self.attRefOutMsg.write(refMsg, CurrentSimNanos)


# pre-compute initial attitude so the controller starts near the target
# and never has to traverse a large MRP arc at t=0 (that would spike the torque)
sun_hat = approx_sun_hat_from_epoch(SIM_EPOCH_UTC)

mu_init = 3.986004415e14
oe_init = orbitalMotion.ClassicElements()
oe_init.a = (6371.0 + 400.0) * 1000.0  # 400 km circular
oe_init.e = 0.0001; oe_init.i = 51.6 * macros.D2R  # ISS-like inclination
oe_init.Omega = 0.0; oe_init.omega = 0.0; oe_init.f = 90.0 * macros.D2R
rN_init, _ = orbitalMotion.elem2rv(mu_init, oe_init)
earth_hat_init = -np.array(rN_init) / np.linalg.norm(rN_init)

if ADCS_MODE == "ROLL_ONLY":
    x_B_init = -sun_hat  # same logic as guidance: -X toward Sun
else:
    rho_rad  = np.arcsin(6371.0 / (6371.0 + 400.0))
    anti_sun = -sun_hat
    u2       = anti_sun - np.dot(anti_sun, earth_hat_init) * earth_hat_init
    u2      /= np.linalg.norm(u2)
    x_B_init = np.cos(rho_rad) * earth_hat_init + np.sin(rho_rad) * u2  # limb-pointing

# build a right-handed frame and convert to initial MRP
ref_vec   = np.array([0, 0, 1]) if abs(np.dot(x_B_init, [0, 0, 1])) < 0.9 else np.array([0, 1, 0])
z_B_init  = np.cross(x_B_init, ref_vec);  z_B_init /= np.linalg.norm(z_B_init)
y_B_init  = np.cross(z_B_init, x_B_init); y_B_init /= np.linalg.norm(y_B_init)
sigma_BN_init = rbk.C2MRP(np.array([x_B_init, y_B_init, z_B_init]))
print(f"Initial sigma_BN (near target): {sigma_BN_init.tolist()}")


scSim       = SimulationBaseClass.SimBaseClass()
simTaskName = "dynamicsTask"
dynProcess  = scSim.CreateNewProcess("dynamicsProcess")


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

guidance          = ActiveGuidance(mode=ADCS_MODE, epoch_iso_utc=SIM_EPOCH_UTC)
guidance.ModelTag = "activeGuidance"
guidance.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

# wire up the SPICE Sun message so guidance gets a real-time Sun direction
sun_index = 1
if hasattr(gravFactory, "spicePlanetNames") and "sun" in gravFactory.spicePlanetNames:
    sun_index = gravFactory.spicePlanetNames.index("sun")
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

viz = vizSupport.enableUnityVisualization(
    scSim, simTaskName, scObject,
    saveFile=save_path,
    oscOrbitColorList=[[80, 180, 255, 255]],
    trueOrbitColorList=[[255, 255, 255, 180]],
)
assert viz is not None, "Vizard setup failed"

viz.settings.orbitLinesOn                    = 1
viz.settings.trueTrajectoryLinesOn           = 1
viz.settings.spacecraftCSon                  = 1
viz.settings.showSpacecraftLabels            = 1
viz.settings.showCelestialBodyLabels         = 1
viz.settings.showCameraLabels                = 1
viz.settings.forceStartAtSpacecraftLocalView = 1
viz.settings.spacecraftSizeMultiplier        = 1.0
viz.settings.viewCameraBoresightHUD          = 1
viz.settings.viewCameraConeHUD               = 1



# force a simple visible mesh so spacecraft doesn't appear as a dark placeholder
vizSupport.createCustomModel(viz,
    modelPath="CUBE",
    scale=[BODY_LENGTH_X_M, BODY_BASE_YZ_M, BODY_BASE_YZ_M],
    simBodiesToModify=[scObject.ModelTag])

vizSupport.createPointLine(viz, toBodyName="earth", lineColor=[0, 180, 255, 200])
vizSupport.createPointLine(viz, toBodyName="sun",   lineColor=[255, 220,   0, 200])

RED    = [220,  30,  30, 230]
ORANGE = [255, 140,   0, 180]

for body in ["earth", "sun", "moon"]:
    # outer warning cone (orange): objects should remain outside
    vizSupport.createConeInOut(viz,
        fromBodyName=scObject.ModelTag, toBodyName=body,
        coneColor=ORANGE, normalVector_B=VEC_LOST_B, position_B=POS_LOST_B,
        incidenceAngle=LOST_EXCL_HALF_DEG * macros.D2R, isKeepIn=False, coneHeight=100.0,
        coneName=f"LOST_{body}_EXCL")
    # inner red cone: strict LOST keep-out region
    vizSupport.createConeInOut(viz,
        fromBodyName=scObject.ModelTag, toBodyName=body,
        coneColor=RED, normalVector_B=VEC_LOST_B, position_B=POS_LOST_B,
        incidenceAngle=LOST_HALF_DEG * macros.D2R, isKeepIn=False, coneHeight=100.0,
        coneName=f"LOST_{body}_FOV")

vizSupport.createConeInOut(viz,
    fromBodyName=scObject.ModelTag, toBodyName="earth",
    coneColor=RED, normalVector_B=VEC_FOUND_B, position_B=POS_FOUND_B,
    incidenceAngle=FOUND_HALF_DEG * macros.D2R, isKeepIn=True, coneHeight=100.0,
    coneName="FOUND_Earth_FOV")
vizSupport.createConeInOut(viz,
    fromBodyName=scObject.ModelTag, toBodyName="sun",
    coneColor=ORANGE, normalVector_B=VEC_FOUND_B, position_B=POS_FOUND_B,
    incidenceAngle=FOUND_EXCL_HALF_DEG * macros.D2R, isKeepIn=False, coneHeight=100.0,
    coneName="FOUND_Sun_EXCL")
vizSupport.createConeInOut(viz,
    fromBodyName=scObject.ModelTag, toBodyName="sun",
    coneColor=RED, normalVector_B=VEC_FOUND_B, position_B=POS_FOUND_B,
    incidenceAngle=FOUND_HALF_DEG * macros.D2R, isKeepIn=False, coneHeight=100.0,
    coneName="FOUND_Sun_FOV")

vizSupport.createStandardCamera(viz,
    setMode=1, bodyTarget="cubesat",
    fieldOfView=LOST_FOV_DEG * macros.D2R, pointingVector_B=VEC_LOST_B,
    position_B=POS_LOST_B, displayName="LOST_cam")
vizSupport.createStandardCamera(viz,
    setMode=1, bodyTarget="cubesat",
    fieldOfView=FOUND_FOV_DEG * macros.D2R, pointingVector_B=VEC_FOUND_B,
    position_B=POS_FOUND_B, displayName="FOUND_cam")

print(f"Mode: {ADCS_MODE}")
scSim.InitializeSimulation()
scSim.ConfigureStopTime(macros.hour2nano(SIM_HOURS))
scSim.ExecuteSimulation()
print(f"Done! Load {save_path} in Vizard.")

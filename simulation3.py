import os
import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, orbitalMotion, vizSupport, RigidBodyKinematics as rbk
from Basilisk.simulation import spacecraft, extForceTorque, simpleNav
from Basilisk.fswAlgorithms import attTrackingError, mrpPD
from Basilisk.architecture import sysModel, messaging


# toggle
# "ROLL_ONLY"  -> Lock -X to Sun, optimizes for lost
# "COMPROMISE" -> FOUND (+X) tracks Earth limb with rho-offset, kinda seems pretty good for both lost/found?
ADCS_MODE = "ROLL_ONLY"

# class to help fix attitude (kinda simulate ADCS)
class ActiveGuidance(sysModel.SysModel):
    def __init__(self, mode="ROLL_ONLY"):
        super().__init__()
        self.mode = mode
        self.attRefOutMsg = messaging.AttRefMsg()
        self.scStateInMsg = messaging.SCStatesMsgReader()
        self.prev_roll_deg = None

        lon = np.radians(39.0)
        obl = np.radians(23.4)
        self.sun_hat = np.array([np.cos(lon),
                                  np.sin(lon) * np.cos(obl),
                                  np.sin(lon) * np.sin(obl)])
        self.sun_hat /= np.linalg.norm(self.sun_hat)

        self.rho_rad = np.arcsin(6371.0 / (6371.0 + 400.0))

    def Reset(self, CurrentSimNanos):
        pass

    def UpdateState(self, CurrentSimNanos):
        scState = self.scStateInMsg()
        r_N     = np.array(scState.r_BN_N)
        r_mag   = np.linalg.norm(r_N)

        # Guard against uninitialized message at t=0
        if r_mag < 1.0:
            refMsg             = messaging.AttRefMsgPayload()
            refMsg.sigma_RN    = [0.0, 0.0, 0.0]
            refMsg.omega_RN_N  = [0.0, 0.0, 0.0]
            refMsg.domega_RN_N = [0.0, 0.0, 0.0]
            self.attRefOutMsg.write(refMsg, CurrentSimNanos)
            return

        earth_hat = -r_N / r_mag

        if self.mode == "ROLL_ONLY":
            x_B = -self.sun_hat

        elif self.mode == "COMPROMISE":
            u1       = earth_hat
            anti_sun = -self.sun_hat
            u2       = anti_sun - np.dot(anti_sun, u1) * u1
            n2       = np.linalg.norm(u2)
            if n2 < 1e-6:
                u2 = np.array([1, 0, 0]) if abs(u1[0]) < 0.9 else np.array([0, 1, 0])
                u2 = u2 - np.dot(u2, u1) * u1
                n2 = np.linalg.norm(u2)
            u2 /= n2
            x_B = np.cos(self.rho_rad) * u1 + np.sin(self.rho_rad) * u2

        x_B /= np.linalg.norm(x_B)

        z_temp = np.cross(x_B, earth_hat)
        if np.linalg.norm(z_temp) < 1e-6:
            z_temp = np.cross(x_B, np.array([0, 1, 0]))
            if np.linalg.norm(z_temp) < 1e-6:
                z_temp = np.cross(x_B, np.array([0, 0, 1]))
        z_temp /= np.linalg.norm(z_temp)
        y_temp  = np.cross(z_temp, x_B)
        y_temp /= np.linalg.norm(y_temp)

        roll_candidates = []
        max_score = -1.0

        for roll_deg in range(0, 360, 2):
            roll     = np.radians(roll_deg)
            y_test   = np.cos(roll) * y_temp + np.sin(roll) * z_temp
            z_test   = np.cross(x_B, y_test)
            z_test  /= np.linalg.norm(z_test)
            lost_hat = -z_test

            ang_earth = np.degrees(np.arccos(np.clip(np.dot(lost_hat, earth_hat),    -1.0, 1.0)))
            ang_sun   = np.degrees(np.arccos(np.clip(np.dot(lost_hat, self.sun_hat), -1.0, 1.0)))
            score     = min(ang_earth, ang_sun)

            roll_candidates.append((roll_deg, score, y_test, z_test))
            if score > max_score:
                max_score = score

        keep_margin_deg = 0.25
        near_opt = [c for c in roll_candidates if c[1] >= max_score - keep_margin_deg]

        if self.prev_roll_deg is None:
            best_roll_deg, _, best_y, best_z = max(near_opt, key=lambda c: c[1])
        else:
            def wrap_delta_deg(a, b):
                return abs(((a - b + 180.0) % 360.0) - 180.0)

            best_roll_deg, _, best_y, best_z = min(
                near_opt,
                key=lambda c: wrap_delta_deg(c[0], self.prev_roll_deg)
            )

        self.prev_roll_deg = best_roll_deg

        dcm_RN   = np.array([x_B, best_y, best_z])
        sigma_RN = rbk.C2MRP(dcm_RN)

        # Safety: if C2MRP produced NaN, hold identity
        if np.any(np.isnan(sigma_RN)):
            sigma_RN = np.array([0.0, 0.0, 0.0])

        refMsg             = messaging.AttRefMsgPayload()
        refMsg.sigma_RN    = sigma_RN.tolist()
        refMsg.omega_RN_N  = [0.0, 0.0, 0.0]
        refMsg.domega_RN_N = [0.0, 0.0, 0.0]
        self.attRefOutMsg.write(refMsg, CurrentSimNanos)


lon = np.radians(39.0)
obl = np.radians(23.4)
sun_hat = np.array([np.cos(lon), np.sin(lon)*np.cos(obl), np.sin(lon)*np.sin(obl)])
sun_hat /= np.linalg.norm(sun_hat)

mu_init = 3.986004415e14
oe_init = orbitalMotion.ClassicElements()
oe_init.a = (6371.0 + 400.0) * 1000.0
oe_init.e = 0.0001; oe_init.i = 51.6 * macros.D2R
oe_init.Omega = 0.0; oe_init.omega = 0.0; oe_init.f = 90.0 * macros.D2R
rN_init, _ = orbitalMotion.elem2rv(mu_init, oe_init)
earth_hat_init = -np.array(rN_init) / np.linalg.norm(rN_init)

if ADCS_MODE == "ROLL_ONLY":
    x_B_init = -sun_hat
else:
    rho_rad  = np.arcsin(6371.0 / (6371.0 + 400.0))
    anti_sun = -sun_hat
    u2       = anti_sun - np.dot(anti_sun, earth_hat_init) * earth_hat_init
    u2      /= np.linalg.norm(u2)
    x_B_init = np.cos(rho_rad) * earth_hat_init + np.sin(rho_rad) * u2

ref_vec   = np.array([0, 0, 1]) if abs(np.dot(x_B_init, [0, 0, 1])) < 0.9 else np.array([0, 1, 0])
z_B_init  = np.cross(x_B_init, ref_vec);  z_B_init /= np.linalg.norm(z_B_init)
y_B_init  = np.cross(z_B_init, x_B_init); y_B_init /= np.linalg.norm(y_B_init)
sigma_BN_init = rbk.C2MRP(np.array([x_B_init, y_B_init, z_B_init]))
print(f"Initial sigma_BN (near target): {sigma_BN_init.tolist()}")

scSim       = SimulationBaseClass.SimBaseClass()
simTaskName = "dynamicsTask"
dynProcess  = scSim.CreateNewProcess("dynamicsProcess")

# 0.5 s timestep: ω_n*dt = 0.707*0.5 = 0.35 << π  (stable)
# 92-min sim => 11,040 frames
dynProcess.addTask(scSim.CreateNewTask(simTaskName, macros.sec2nano(0.5)))

scObject          = spacecraft.Spacecraft()
scObject.ModelTag = "cubesat"
scObject.hub.mHub = 12.0
scObject.hub.IHubPntBc_B = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]
scSim.AddModelToTask(simTaskName, scObject)

gravFactory = simIncludeGravBody.gravBodyFactory()
earth = gravFactory.createEarth(); earth.isCentralBody = True
sun   = gravFactory.createSun()
moon  = gravFactory.createMoon()
gravFactory.createSpiceInterface(time="2024-05-01T12:00:00.000Z", epochInMsg=True)
gravFactory.spiceObject.zeroBase = "earth"
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
scObject.hub.sigma_BNInit   = sigma_BN_init.tolist()   # near target, not identity
scObject.hub.omega_BN_BInit = [0.0, 0.0, 0.0]


sNav          = simpleNav.SimpleNav()
sNav.ModelTag = "SimpleNav"
scSim.AddModelToTask(simTaskName, sNav)
sNav.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

guidance          = ActiveGuidance(mode=ADCS_MODE)
guidance.ModelTag = "activeGuidance"
guidance.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, guidance)

attErr          = attTrackingError.attTrackingError()
attErr.ModelTag = "attError"
scSim.AddModelToTask(simTaskName, attErr)
attErr.attNavInMsg.subscribeTo(sNav.attOutMsg)
attErr.attRefInMsg.subscribeTo(guidance.attRefOutMsg)

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
    os.remove(save_path)

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

vec_LOST  = [0, 0, -1]
vec_FOUND = [1, 0,  0]
sensor_offset_m = 0.15
pos_LOST  = [sensor_offset_m * x for x in vec_LOST]
pos_FOUND = [sensor_offset_m * x for x in vec_FOUND]

vizSupport.createCustomModel(viz,
    modelPath="CUBE",
    scale=[0.30, 0.20, 0.10],
    simBodiesToModify=[scObject.ModelTag])

vizSupport.createPointLine(viz, toBodyName="earth", lineColor=[0, 180, 255, 200])
vizSupport.createPointLine(viz, toBodyName="sun",   lineColor=[255, 220,   0, 200])

RED    = [220,  30,  30, 230]
ORANGE = [255, 140,   0, 180]

for body in ["earth", "sun", "moon"]:
    vizSupport.createConeInOut(viz,
        fromBodyName=scObject.ModelTag, toBodyName=body,
        coneColor=ORANGE, normalVector_B=vec_LOST, position_B=pos_LOST,
        incidenceAngle=17.5 * macros.D2R, isKeepIn=False, coneHeight=100.0,
        coneName=f"LOST_{body}_EXCL")
    vizSupport.createConeInOut(viz,
        fromBodyName=scObject.ModelTag, toBodyName=body,
        coneColor=RED, normalVector_B=vec_LOST, position_B=pos_LOST,
        incidenceAngle=7.5 * macros.D2R, isKeepIn=False, coneHeight=100.0,
        coneName=f"LOST_{body}_FOV")

vizSupport.createConeInOut(viz,
    fromBodyName=scObject.ModelTag, toBodyName="earth",
    coneColor=RED, normalVector_B=vec_FOUND, position_B=pos_FOUND,
    incidenceAngle=30.0 * macros.D2R, isKeepIn=True, coneHeight=100.0,
    coneName="FOUND_Earth_FOV")
vizSupport.createConeInOut(viz,
    fromBodyName=scObject.ModelTag, toBodyName="sun",
    coneColor=ORANGE, normalVector_B=vec_FOUND, position_B=pos_FOUND,
    incidenceAngle=40.0 * macros.D2R, isKeepIn=False, coneHeight=100.0,
    coneName="FOUND_Sun_EXCL")
vizSupport.createConeInOut(viz,
    fromBodyName=scObject.ModelTag, toBodyName="sun",
    coneColor=RED, normalVector_B=vec_FOUND, position_B=pos_FOUND,
    incidenceAngle=30.0 * macros.D2R, isKeepIn=False, coneHeight=100.0,
    coneName="FOUND_Sun_FOV")

vizSupport.createStandardCamera(viz,
    setMode=1, bodyTarget="cubesat",
    fieldOfView=15.0 * macros.D2R, pointingVector_B=vec_LOST,
    position_B=pos_LOST, displayName="LOST_cam")
vizSupport.createStandardCamera(viz,
    setMode=1, bodyTarget="cubesat",
    fieldOfView=60.0 * macros.D2R, pointingVector_B=vec_FOUND,
    position_B=pos_FOUND, displayName="FOUND_cam")


print(f"Mode: {ADCS_MODE}")
scSim.InitializeSimulation()
scSim.ConfigureStopTime(macros.hour2nano(24))
scSim.ExecuteSimulation()
print(f"Done! Load {save_path} in Vizard.")
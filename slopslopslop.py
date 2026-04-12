import os
import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, orbitalMotion, vizSupport, RigidBodyKinematics as rbk
from Basilisk.simulation import spacecraft, extForceTorque, simpleNav
from Basilisk.fswAlgorithms import attTrackingError, mrpPD
from Basilisk.architecture import sysModel, messaging


ADCS_MODE = "ROLL_ONLY"

class ActiveGuidance(sysModel.SysModel):
    def __init__(self, mode="COMPROMISE"):
        super().__init__()
        self.mode = mode
        self.attRefOutMsg = messaging.AttRefMsg()
        self.scStateInMsg = messaging.SCStatesMsgReader()

        # i think this for 2025-04-01
        lon  = np.radians(39.0)
        obl  = np.radians(23.4)
        self.sun_hat = np.array([np.cos(lon),
                                  np.sin(lon) * np.cos(obl),
                                  np.sin(lon) * np.sin(obl)])
        self.sun_hat /= np.linalg.norm(self.sun_hat)

        self.rho_rad = np.radians(70.2)

    def Reset(self, CurrentSimNanos):
        pass

    def UpdateState(self, CurrentSimNanos):
        scState  = self.scStateInMsg()
        r_N      = np.array(scState.r_BN_N)
        earth_hat = -r_N / np.linalg.norm(r_N)

        if self.mode == "COMPROMISE":
            u1 = earth_hat
            anti_sun = -self.sun_hat
            u2 = anti_sun - np.dot(anti_sun, u1) * u1
            n2 = np.linalg.norm(u2)
            if n2 < 1e-6:
                u2 = np.array([1, 0, 0]) if abs(u1[0]) < 0.9 else np.array([0, 1, 0])
                u2 = u2 - np.dot(u2, u1) * u1
                n2 = np.linalg.norm(u2)
            u2 /= n2
            x_B = np.cos(self.rho_rad) * u1 + np.sin(self.rho_rad) * u2

        elif self.mode == "ROLL_ONLY":
            x_B = -self.sun_hat


        z_temp = np.cross(x_B, earth_hat)
        if np.linalg.norm(z_temp) < 1e-6:
            z_temp = np.array([0, 0, 1])
        z_temp /= np.linalg.norm(z_temp)
        y_temp = np.cross(z_temp, x_B)

        best_y, best_z = y_temp, z_temp
        max_score = -1.0

        for roll_deg in range(0, 360, 5):
            roll    = np.radians(roll_deg)
            y_test  = np.cos(roll) * y_temp + np.sin(roll) * z_temp
            z_test  = np.cross(x_B, y_test)
            lost_hat = -z_test

            ang_earth = np.degrees(np.arccos(np.clip(np.dot(lost_hat, earth_hat),  -1, 1)))
            ang_sun   = np.degrees(np.arccos(np.clip(np.dot(lost_hat, self.sun_hat), -1, 1)))
            score = min(ang_earth, ang_sun)

            if score > max_score:
                max_score = score
                best_y, best_z = y_test, z_test

        dcm_RN   = np.array([x_B, best_y, best_z])
        sigma_RN = rbk.C2MRP(dcm_RN)

        refMsg             = messaging.AttRefMsgPayload()
        refMsg.sigma_RN    = sigma_RN
        refMsg.omega_RN_N  = [0.0, 0.0, 0.0]
        refMsg.domega_RN_N = [0.0, 0.0, 0.0]
        self.attRefOutMsg.write(refMsg, CurrentSimNanos)


scSim       = SimulationBaseClass.SimBaseClass()
simTaskName = "dynamicsTask"
dynProcess  = scSim.CreateNewProcess("dynamicsProcess")
dynProcess.addTask(scSim.CreateNewTask(simTaskName, macros.sec2nano(1.0)))

scObject           = spacecraft.Spacecraft()
scObject.ModelTag  = "cubesat"
scObject.hub.mHub  = 12.0
scObject.hub.IHubPntBc_B = [[0.1, 0, 0],
                             [0, 0.1, 0],
                             [0, 0, 0.1]]
scSim.AddModelToTask(simTaskName, scObject)

gravFactory = simIncludeGravBody.gravBodyFactory()
earth = gravFactory.createEarth()
earth.isCentralBody = True
sun  = gravFactory.createSun()
moon = gravFactory.createMoon()
gravFactory.createSpiceInterface(time="2024-05-01T12:00:00.000Z", epochInMsg=True)
gravFactory.spiceObject.zeroBase = "earth"
scSim.AddModelToTask(simTaskName, gravFactory.spiceObject)
scObject.gravField.gravBodies = spacecraft.GravBodyVector(
    list(gravFactory.gravBodies.values()))

mu    = 3.986004415e14
oe    = orbitalMotion.ClassicElements()
oe.a  = (6371.0 + 400.0) * 1000.0
oe.e  = 0.0001
oe.i  = 51.6 * macros.D2R
oe.Omega = 0.0
oe.omega = 0.0
oe.f     = 90.0 * macros.D2R
rN, vN   = orbitalMotion.elem2rv(mu, oe)

scObject.hub.r_CN_NInit      = rN
scObject.hub.v_CN_NInit      = vN
scObject.hub.sigma_BNInit    = [0.0, 0.0, 0.0]
scObject.hub.omega_BN_BInit  = [0.0, 0.0, 0.0]


guidance          = ActiveGuidance(mode=ADCS_MODE)
guidance.ModelTag = "customGuidance"
guidance.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, guidance)

sNavObject          = simpleNav.SimpleNav()
sNavObject.ModelTag = "SimpleNav"
scSim.AddModelToTask(simTaskName, sNavObject)
sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

attError          = attTrackingError.attTrackingError()
attError.ModelTag = "attError"
scSim.AddModelToTask(simTaskName, attError)
attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
attError.attRefInMsg.subscribeTo(guidance.attRefOutMsg)

mrpControl          = mrpPD.mrpPD()
mrpControl.ModelTag = "mrpPD"
mrpControl.K        = 1.5
mrpControl.P        = 2.5
scSim.AddModelToTask(simTaskName, mrpControl)
mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)

vehicleConfig               = messaging.VehicleConfigMsgPayload()
vehicleConfig.ISCPntB_B     = [0.1, 0.0, 0.0,
                               0.0, 0.1, 0.0,
                               0.0, 0.0, 0.1]
vehConfigMsg = messaging.VehicleConfigMsg().write(vehicleConfig)
mrpControl.vehConfigInMsg.subscribeTo(vehConfigMsg)

extFT          = extForceTorque.ExtForceTorque()
extFT.ModelTag = "extFT"
scSim.AddModelToTask(simTaskName, extFT)
scObject.addDynamicEffector(extFT)
extFT.cmdTorqueInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)

viz = vizSupport.enableUnityVisualization(
    scSim,
    simTaskName,
    scObject,
    saveFile=os.path.join(os.getcwd(), f'simulation_{ADCS_MODE}.bin')
)

viz.settings.skyBox                      = "Black"
viz.settings.orbitLinesOn                = 1
viz.settings.trueTrajectoryLinesOn       = 1
viz.settings.spacecraftCSon              = 1
viz.settings.planetCSon                  = 1
viz.settings.showSpacecraftLabels        = 1
viz.settings.showCelestialBodyLabels     = 1
viz.settings.showCameraLabels            = 1
viz.settings.forceStartAtSpacecraftLocalView = 0
viz.settings.spacecraftSizeMultiplier    = 10.0
viz.settings.viewCameraBoresightHUD      = 1
viz.settings.viewCameraConeHUD           = 1
vizSupport.createPointLine(viz, toBodyName="earth", lineColor=[0, 180, 255, 200])
vizSupport.createPointLine(viz, toBodyName="sun",   lineColor=[255, 220, 0, 200])

RED    = [220, 30,  30,  230]
ORANGE = [255, 140,  0,  180]

found_half_rad = 30.0 * macros.D2R
lost_half_rad  =  7.5 * macros.D2R
excl_half_rad  = 17.5 * macros.D2R

vec_LOST  = [0, 0, -1]
vec_FOUND = [1, 0,  0]
sensor_offset_m = 0.15
pos_LOST  = [sensor_offset_m * x for x in vec_LOST]
pos_FOUND = [sensor_offset_m * x for x in vec_FOUND]

for body in ["earth", "sun", "moon"]:
    vizSupport.createConeInOut(viz,
        fromBodyName=scObject.ModelTag, toBodyName=body,
        coneColor=ORANGE, normalVector_B=vec_LOST, position_B=pos_LOST,
        incidenceAngle=excl_half_rad, isKeepIn=False,
        coneHeight=100.0, coneName=f"LOST_{body}_EXCL")

    vizSupport.createConeInOut(viz,
        fromBodyName=scObject.ModelTag, toBodyName=body,
        coneColor=RED, normalVector_B=vec_LOST, position_B=pos_LOST,
        incidenceAngle=lost_half_rad, isKeepIn=False,
        coneHeight=100.0, coneName=f"LOST_{body}_FOV")

vizSupport.createConeInOut(viz,
    fromBodyName=scObject.ModelTag, toBodyName="earth",
    coneColor=RED, normalVector_B=vec_FOUND, position_B=pos_FOUND,
    incidenceAngle=found_half_rad, isKeepIn=True,
    coneHeight=100.0, coneName="FOUND_Earth_FOV")


vizSupport.createConeInOut(viz,
    fromBodyName=scObject.ModelTag, toBodyName="sun",
    coneColor=ORANGE, normalVector_B=vec_FOUND, position_B=pos_FOUND,
    incidenceAngle=excl_half_rad, isKeepIn=False,
    coneHeight=100.0, coneName="FOUND_Sun_EXCL")
vizSupport.createConeInOut(viz,
    fromBodyName=scObject.ModelTag, toBodyName="sun",
    coneColor=RED, normalVector_B=vec_FOUND, position_B=pos_FOUND,
    incidenceAngle=found_half_rad, isKeepIn=False,
    coneHeight=100.0, coneName="FOUND_Sun_FOV")

vizSupport.createStandardCamera(viz,
    setMode=1, bodyTarget="cubesat",
    fieldOfView=15.0 * macros.D2R,
    pointingVector_B=vec_LOST,
    position_B=pos_LOST,
    displayName="LOST_cam")

vizSupport.createStandardCamera(viz,
    setMode=1, bodyTarget="cubesat",
    fieldOfView=60.0 * macros.D2R,
    pointingVector_B=vec_FOUND,
    position_B=pos_FOUND,
    displayName="FOUND_cam")

print(f"Running Active ADCS Simulation — Mode: {ADCS_MODE}")
scSim.InitializeSimulation()
scSim.ConfigureStopTime(macros.min2nano(92))
scSim.ExecuteSimulation()
print(f"Done! Load  simulation_{ADCS_MODE}.bin  in Vizard.")

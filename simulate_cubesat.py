import os
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, orbitalMotion, vizSupport
from Basilisk.simulation import spacecraft

scSim = SimulationBaseClass.SimBaseClass()
simTaskName = "dynamicsTask"
simProcessName = "dynamicsProcess"

dynProcess = scSim.CreateNewProcess(simProcessName)
dynProcess.addTask(scSim.CreateNewTask(simTaskName, macros.sec2nano(1.0)))

scObject = spacecraft.Spacecraft()
scObject.ModelTag = "cubesat"
scSim.AddModelToTask(simTaskName, scObject)
gravFactory = simIncludeGravBody.gravBodyFactory()

earth = gravFactory.createEarth()
earth.isCentralBody = True
sun = gravFactory.createSun()
moon = gravFactory.createMoon()

gravFactory.createSpiceInterface(time="2024-05-01T12:00:00.000Z", epochInMsg=True)
gravFactory.spiceObject.zeroBase = "earth"

scSim.AddModelToTask(simTaskName, gravFactory.spiceObject)
scObject.gravField.gravBodies = spacecraft.GravBodyVector(list(gravFactory.gravBodies.values()))

mu = earth.mu
oe = orbitalMotion.ClassicElements()
# leo orbit I think ? maybe :(
oe.a = (6371.0 + 400.0) * 1000.0
oe.e = 0.0001
oe.i = 51.6 * macros.D2R
oe.Omega = 0.0
oe.omega = 0.0
oe.f = 0.0

rN, vN = orbitalMotion.elem2rv(mu, oe)
scObject.hub.r_CN_NInit = rN
scObject.hub.v_CN_NInit = vN


# initial MRP attitude
ATTITUDE_SIGMA_BN = [0.0, 0.0, 0.0]
# rad/s body rate
ATTITUDE_RATE_BN_B = [0.02, 0.06, 0.03]

scObject.hub.sigma_BNInit = [[ATTITUDE_SIGMA_BN[0]], [ATTITUDE_SIGMA_BN[1]], [ATTITUDE_SIGMA_BN[2]]]
scObject.hub.omega_BN_BInit = [[ATTITUDE_RATE_BN_B[0]], [ATTITUDE_RATE_BN_B[1]], [ATTITUDE_RATE_BN_B[2]]]

viz = vizSupport.enableUnityVisualization(
    scSim,
    simTaskName,
    scObject,
    saveFile=os.path.join(os.getcwd(), "output.bin")
)
viz.settings.skyBox = "Black"

# CHANGE FOVs
FOV_LOST = 45.0
FOV_FOUND = 45.0
ERROR_MARGIN = 0.04 # i think this is what josh said

lost_angle = (FOV_LOST / 2.0 + ERROR_MARGIN) * macros.D2R
found_angle = (FOV_FOUND / 2.0 + ERROR_MARGIN) * macros.D2R

# stupid freakin colors
CONE_GREEN = [0, 255, 0, 180]
CONE_RED = [255, 0, 0, 255]
CONE_HEIGHT_CAP_M = 100

# LOST out the top (-Z), FOUND 90 deg away (+X)
vec_LOST = [0, 0, -1]
vec_FOUND = [1, 0, 0]

# Offset cameras/cones 1m cause of satellite model
cam_offset = 0.908
cam_offset2 = 1
lost_origin_B = [cam_offset2 * x for x in vec_LOST]    # [0, 0, -1]
found_origin_B = [cam_offset * x for x in vec_FOUND]  # [-1, 0, 0]

# we want all bodies outside lost cone
for body in ["earth", "sun", "moon"]:
    # if green this body is out yay!
    vizSupport.createConeInOut(
        viz,
        fromBodyName=scObject.ModelTag,
        toBodyName=body,
        coneColor=CONE_GREEN,
        normalVector_B=vec_LOST,
        position_B=lost_origin_B,
        incidenceAngle=lost_angle,
        isKeepIn=True,
        coneHeight=CONE_HEIGHT_CAP_M,
        coneName=f"LOST_{body}_GREEN"
    )
    
    # if red nooo its there D:
    vizSupport.createConeInOut(
        viz,
        fromBodyName=scObject.ModelTag,
        toBodyName=body,
        coneColor=CONE_RED,
        normalVector_B=vec_LOST,
        position_B=lost_origin_B,
        incidenceAngle=lost_angle,
        isKeepIn=False,
        coneHeight=CONE_HEIGHT_CAP_M,
        coneName=f"LOST_{body}_RED"
    )

# found is weird
# green if earth there
vizSupport.createConeInOut(
    viz,
    fromBodyName=scObject.ModelTag,
    toBodyName="earth",
    coneColor=CONE_GREEN,
    normalVector_B=vec_FOUND,
    position_B=found_origin_B,
    incidenceAngle=found_angle,
    isKeepIn=False,
    coneHeight=CONE_HEIGHT_CAP_M,
    coneName="FOUND_Earth_GREEN"
)

# red if earth not there i think?
vizSupport.createConeInOut(
    viz,
    fromBodyName=scObject.ModelTag,
    toBodyName="earth",
    coneColor=CONE_RED,
    normalVector_B=vec_FOUND,
    position_B=found_origin_B,
    incidenceAngle=found_angle,
    isKeepIn=True,
    coneHeight=CONE_HEIGHT_CAP_M,
    coneName="FOUND_Earth_RED"
)

# bad if we have sun
# TODO: cehck on this
vizSupport.createConeInOut(
    viz,
    fromBodyName=scObject.ModelTag,
    toBodyName="sun",
    coneColor=CONE_RED,
    normalVector_B=vec_FOUND,
    position_B=found_origin_B,
    incidenceAngle=found_angle,
    isKeepIn=False,
    coneHeight=CONE_HEIGHT_CAP_M,
    coneName="FOUND_Sun_RED"
)

# lost camera - has same offset as cone
vizSupport.createStandardCamera(
    viz,
    setMode=1, # body-fixed camera mode
    bodyTarget="cubesat",
    fieldOfView=FOV_LOST * macros.D2R,
    pointingVector_B=vec_LOST,
    position_B=lost_origin_B,
    displayName="LOST_cam"
)

# found camera has same offset as cone
vizSupport.createStandardCamera(
    viz,
    setMode=1, # body-fixed camera mode
    bodyTarget="cubesat",
    fieldOfView=FOV_FOUND * macros.D2R,
    pointingVector_B=vec_FOUND,
    position_B=found_origin_B,
    displayName="FOUND_cam"
)

scSim.InitializeSimulation()
scSim.ConfigureStopTime(macros.hour2nano(24)) # 24hr sim
scSim.ExecuteSimulation()

print("Compilation worked :D")

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

scObject.gravField.gravBodies = spacecraft.GravBodyVector(list(gravFactory.gravBodies.values()))
# scObject.gravField.gravityBodies = spacecraft.GravityBodyVector(list(gravFactory.gravBodies.values()))

mu = earth.mu
oe = orbitalMotion.ClassicElements()
oe.a = (6371.0 + 400.0) * 1000.0
oe.e = 0.0001
oe.i = 51.6 * macros.D2R
oe.Omega = 0.0
oe.omega = 0.0
oe.f = 0.0

rN, vN = orbitalMotion.elem2rv(mu, oe)
scObject.hub.r_CN_NInit = rN
scObject.hub.v_CN_NInit = vN
scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
scObject.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]

viz = vizSupport.enableUnityVisualization(
    scSim,
    simTaskName,
    scObject,
    saveFile=os.path.join(os.getcwd(), "simulation1.bin")
)

# LOST out the top
vizSupport.createConeInOut(
    viz,
    fromBodyName=scObject.ModelTag,
    toBodyName="earth",
    coneColor="red",
    normalVector_B=[0, 0, 1],
    incidenceAngle=(20.0 / 2.0 + 0.04) * macros.D2R,
    isKeepIn=False,
    coneHeight=1.0,
    coneName="LOST_earth_KO"
)

scSim.InitializeSimulation()
scSim.ConfigureStopTime(macros.min2nano(5.0))
scSim.ExecuteSimulation()

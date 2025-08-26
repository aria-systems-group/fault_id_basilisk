from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion, unitTestSupport
from Basilisk.simulation import spacecraft
from Basilisk.utilities import simIncludeGravBody
from Basilisk.utilities import (macros, simIncludeRW, unitTestSupport)
from Basilisk.architecture import messaging
from Basilisk.simulation import reactionWheelStateEffector


def setup_spacecraft_sim(
    true_mode=0,
    simTimeSec=600,
    simTimeStepSec=0.1,
    # --- Initial State Inputs ---
    oe=None,                  # Classical orbital elements
    rN=None,                  # Initial position vector (m)
    vN=None,                  # Initial velocity vector (m/s)
    sigma_BNInit=None,        # Initial MRPs
    omega_BN_BInit=None       # Initial body rates (rad/s)
):
    """
    Setup spacecraft simulation with configurable initial state.
    If rN and vN are not provided, they are derived from orbital elements.
    """

    # --- Create Simulation ---
    simTaskName = "simTask"
    simProcessName = "simProcess"
    scSim = SimulationBaseClass.SimBaseClass()
    simulationTime = macros.sec2nano(simTimeSec)
    simulationTimeStep = macros.sec2nano(simTimeStepSec)

    dynProcess = scSim.CreateNewProcess(simProcessName)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # --- Setup Gravity ---
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    earth.isCentralBody = True
    mu = earth.mu

    # --- Create Spacecraft ---
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bsk-Sat"

    I = [900., 0., 0.,
         0., 800., 0.,
         0., 0., 600.]
    scObject.hub.mHub = 750.0
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)

    # --- Default orbital elements if none given ---
    if oe is None:
        oe = orbitalMotion.ClassicElements()
        oe.a = 10000000.0
        oe.e = 0.01
        oe.i = 33.3 * macros.D2R
        oe.Omega = 48.2 * macros.D2R
        oe.omega = 347.8 * macros.D2R
        oe.f = 85.3 * macros.D2R

    # --- Compute position/velocity if not provided ---
    if rN is None or vN is None:
        rN, vN = orbitalMotion.elem2rv(mu, oe)

    # --- Apply initial state ---
    scObject.hub.r_CN_NInit = rN
    scObject.hub.v_CN_NInit = vN
    scObject.hub.sigma_BNInit = sigma_BNInit if sigma_BNInit is not None else [[0.1], [0.2], [-0.3]]
    scObject.hub.omega_BN_BInit = omega_BN_BInit if omega_BN_BInit is not None else [[0.0], [0.0], [0.0]]

    scSim.AddModelToTask(simTaskName, scObject, 1)
    gravFactory.addBodiesTo(scObject)

    # --- Setup Reaction Wheels ---
    rwFactory = simIncludeRW.rwFactory()
    varRWModel = messaging.BalancedWheels

    rw_List = []
    for i, gsHat in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        omega_init = 100. + 100. * i
        rWB_B = [0.5, 0.5, 0.5] if i == 2 else None
        rw = rwFactory.create(
            'Honeywell_HR16',
            gsHat,
            maxMomentum=50.,
            Omega=omega_init,
            RWModel=varRWModel,
            rWB_B=rWB_B
        ) if rWB_B else rwFactory.create(
            'Honeywell_HR16',
            gsHat,
            maxMomentum=50.,
            Omega=omega_init,
            RWModel=varRWModel
        )

        # rw_List.append(rw)
        rw_List.append(rw)

    # --- Inject RW fault based on true_mode ---
    if true_mode == 1:
        rw_List[0].u_max = rw_List[0].u_max*0.01
        print("RW1 uMax:", rw_List[0].u_max)
    elif true_mode == 2:
        rw_List[1].u_max = rw_List[1].u_max*0.01
        print("RW2 uMax:", rw_List[1].u_max)
    elif true_mode == 3:
        rw_List[2].u_max = rw_List[2].u_max*0.01
        print("RW3 uMax:", rw_List[2].u_max)

    numRW = rwFactory.getNumOfDevices()

    # --- Add RW effector ---
    rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwStateEffector.ModelTag = "RW_cluster"
    rwFactory.addToSpacecraft(scObject.ModelTag, rwStateEffector, scObject)
    scSim.AddModelToTask(simTaskName, rwStateEffector, 2)


    return (
        scSim,
        scObject,
        simTaskName,
        simTimeSec,
        simTimeStepSec,
        simulationTime,
        simulationTimeStep,
        varRWModel,
        rwFactory,
        rwStateEffector,
        numRW,
        I,
    )

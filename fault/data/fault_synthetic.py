import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import h5py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib.pyplot as plt  # kept for compatibility with existing imports
from Basilisk import __path__
from Basilisk.architecture import messaging
from Basilisk.utilities import macros, unitTestSupport, simIncludeRW, orbitalMotion, simIncludeGravBody
from Basilisk.fswAlgorithms import inertialUKF

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

from utils.spacecraft import setup_spacecraft_sim
from utils.navigation import setup_navigation_and_control
from ukf import configure_inertialattfilter
from utils.messages import setup_messages
from utils.log import setup_logging, process_filter

# ---------------------------
# NEW: Healthy RW dataset generator (no faults)
# ---------------------------
def run_healthy_rw_dataset(
        sample_rate_hz=1,
        duration=50,
        seed: int = 0,
        last_state=None
    ):
    """
    Generates a HEALTHY dataset (no faults) for reaction wheels.
    Logs attitude, body rates, RW speeds, torques, and timestamps.
    Also returns the last spacecraft state for chaining into the next sim phase.
    """

    # --- Set seed for reproducibility ---
    np.random.seed(seed)

    # --- Extract last state if provided ---
    if last_state is not None:
        oe_init = last_state["oe"]
        rN_init = last_state["rN"]
        vN_init = last_state["vN"]
        sigma_init = last_state["sigma"]
        omega_init = last_state["omega"]
    else:
        oe_init = rN_init = vN_init = sigma_init = omega_init = None

    # --- Setup spacecraft ---
    (scSim, scObject, simTaskName, simTimeSec, simTimeStepSec,
     simulationTime, simulationTimeStep, varRWModel, rwFactory,
     rwStateEffector, numRW, I) = setup_spacecraft_sim(
         true_mode=0,               # healthy
         oe=oe_init,                # previous orbital elements (if any)
         rN=rN_init,                # previous position
         vN=vN_init,                # previous velocity
         sigma_BNInit=sigma_init,   # previous attitude MRPs
         omega_BN_BInit=omega_init  # previous body rates
     )

    # Setup navigation + control
    sNavObject, inertial3DObj, attError, mrpControl = \
        setup_navigation_and_control(scSim, simTaskName)

    # Setup messages
    vcMsg, attitude_measurement_msg, st_cov, rwMotorTorqueObj, st_1_data = \
        setup_messages(scSim, simTaskName, I, rwFactory, scObject,
                       sNavObject, attError, inertial3DObj, mrpControl,
                       rwStateEffector)

    fswRwParamMsg = rwFactory.getConfigMessage()
    mrpControl.rwParamsInMsg.subscribeTo(fswRwParamMsg)
    rwMotorTorqueObj.rwParamsInMsg.subscribeTo(fswRwParamMsg)

    # --- Setup inertialUKF (nominal filter) ---
    inertialAttFilter0 = inertialUKF.inertialUKF()
    scSim.AddModelToTask(simTaskName, inertialAttFilter0)

    rwFactory_0 = simIncludeRW.rwFactory()
    rwFactory_0.create('Honeywell_HR16', [1, 0, 0], maxMomentum=50., Omega=100., RWModel=varRWModel)
    rwFactory_0.create('Honeywell_HR16', [0, 1, 0], maxMomentum=50., Omega=200., RWModel=varRWModel)
    rwFactory_0.create('Honeywell_HR16', [0, 0, 1], maxMomentum=50., Omega=300.,
                       rWB_B=[0.5, 0.5, 0.5], RWModel=varRWModel)

    gyroBufferData = messaging.AccDataMsgPayload()
    gyroInMsg = messaging.AccDataMsg()
    gyroInMsg.write(gyroBufferData, 0)

    config0 = {
        "vcMsg": vcMsg,
        "rwStateEffector": rwStateEffector,
        "inertialAttFilterRwParamMsg": rwFactory_0.getConfigMessage(),
        "gyroInMsg": gyroInMsg,
        "st_cov": st_cov,
    }
    configure_inertialattfilter(inertialAttFilter0, config0, attitude_measurement_msg)

    inertialAttFilter0Log = inertialAttFilter0.logger(
        ["covar", "state", "cov_S", "innovation"],
        unitTestSupport.samplingTime(simulationTime, simulationTimeStep, 100)
    )
    scSim.AddModelToTask(simTaskName, inertialAttFilter0Log)

    # Setup logs
    snAttLog, rwLogs = setup_logging(
        scSim, simTaskName,
        unitTestSupport.samplingTime(simulationTime, simulationTimeStep, 100),
        rwMotorTorqueObj, attError, sNavObject, rwStateEffector, numRW
    )

    # --- Simulation duration and sampling ---
    simTimeSec = duration
    dt_sample = 1.0 / sample_rate_hz
    timeSpan = np.arange(0, simTimeSec + dt_sample, dt_sample)

    # Initialize simulation
    scSim.InitializeSimulation()

    # Pre-allocate logs
    att_log, body_rate_log, rw_omega_log, rw_torque_log = [], [], [], []

    # --- Run Simulation loop ---
    for t in timeSpan[1:]:
        scSim.ConfigureStopTime(macros.sec2nano(t))
        scSim.ExecuteSimulation()

        # Write noisy star tracker measurement
        if snAttLog.sigma_BN.shape[0] > 0:
            true_att = snAttLog.sigma_BN[-1, :]
            noisy_att = true_att + np.random.normal(0, np.sqrt(st_cov), 3)
            st_1_data.valid = True
            st_1_data.timeTag = int(t * 1E9)
            st_1_data.MRP_BdyInrtl = noisy_att
            attitude_measurement_msg.write(st_1_data, int(t * 1E9))

        # Logs
        att_log.append(snAttLog.sigma_BN[-1, :])
        body_rate_log.append(snAttLog.omega_BN_B[-1, :])
        rw_omega_log.append([rw.Omega for rw in rwFactory.rwList.values()])
        rw_torque_log.append([
            getattr(rw, "u_current", getattr(rw, "motorTorque", np.nan))
            for rw in rwFactory.rwList.values()
        ])

    # --- Package results ---
    timestamps = timeSpan[1:]
    fault_labels = np.zeros((len(timestamps), numRW), dtype=np.uint8)

    # Get gravitational parameter
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    earth.isCentralBody = True
    mu = earth.mu

    # Use initial hub position/velocity as reference (updated states are in logs)
    rN = np.array(scObject.hub.r_CN_NInit).flatten()
    vN = np.array(scObject.hub.v_CN_NInit).flatten()

    # Compute orbital elements from rN/vN
    oe = orbitalMotion.rv2elem_parab(mu, rN, vN)

    # Last attitude and angular velocity from logs
    sigma = snAttLog.sigma_BN[-1, :]
    omega = snAttLog.omega_BN_B[-1, :]

    last_state_healthy = {
        "oe": oe,
        "rN": rN,
        "vN": vN,
        "sigma": sigma,
        "omega": omega
    }

    dataset_healthy = {
        "timestamps": timestamps,
        "att_log": np.array(att_log),
        "body_rate_log": np.array(body_rate_log),
        "rw_omega_log": np.array(rw_omega_log),
        "rw_torque_log": np.array(rw_torque_log),
        "fault_labels": fault_labels
    }

    return last_state_healthy, dataset_healthy 



def run_faulty_rw_dataset(
    last_state_healthy,
    sample_rate_hz=1,
    duration=50,
    seed: int = 42
):
    """
    Generates a FAULTY dataset (RW1 degraded) for reaction wheels.
    Uses last_state_healthy as input to initialize spacecraft.
    Logs attitude, body rates, RW speeds, torques, and timestamps.
    Returns last spacecraft state for chaining into next phase.
    """

    # --- Set seed ---
    np.random.seed(seed)

    # --- Extract last healthy state ---
    oe_h = last_state_healthy["oe"]
    rN_h = last_state_healthy["rN"]
    vN_h = last_state_healthy["vN"]
    sigma_h = last_state_healthy["sigma"]
    omega_h = last_state_healthy["omega"]

    # --- Setup spacecraft in fault mode using last healthy state ---
    (scSim, scObject, simTaskName, simTimeSec, simTimeStepSec,
     simulationTime, simulationTimeStep, varRWModel, rwFactory,
     rwStateEffector, numRW, I) = setup_spacecraft_sim(
         true_mode=1,   # fault mode
         oe=oe_h,       # Classical orbital elements
         rN=rN_h,       # Initial position vector (m)
         vN=vN_h,       # Initial velocity vector (m/s)
         sigma_BNInit=sigma_h,      # Initial MRPs
         omega_BN_BInit=omega_h     # Initial body rates (rad/s)
     )

    # --- Setup navigation + control ---
    sNavObject, inertial3DObj, attError, mrpControl = \
        setup_navigation_and_control(scSim, simTaskName)

    # --- Setup messages ---
    vcMsg, attitude_measurement_msg, st_cov, rwMotorTorqueObj, st_1_data = \
        setup_messages(scSim, simTaskName, I, rwFactory, scObject,
                       sNavObject, attError, inertial3DObj, mrpControl,
                       rwStateEffector)

    fswRwParamMsg = rwFactory.getConfigMessage()
    mrpControl.rwParamsInMsg.subscribeTo(fswRwParamMsg)
    rwMotorTorqueObj.rwParamsInMsg.subscribeTo(fswRwParamMsg)

    # --- Setup inertialUKF (nominal filter) ---
    inertialAttFilter0 = inertialUKF.inertialUKF()
    scSim.AddModelToTask(simTaskName, inertialAttFilter0)

    rwFactory_0 = simIncludeRW.rwFactory()
    rwFactory_0.create('Honeywell_HR16', [1, 0, 0], maxMomentum=50., Omega=100., RWModel=varRWModel)
    rwFactory_0.create('Honeywell_HR16', [0, 1, 0], maxMomentum=50., Omega=200., RWModel=varRWModel)
    rwFactory_0.create('Honeywell_HR16', [0, 0, 1], maxMomentum=50.,
                       Omega=300., rWB_B=[0.5, 0.5, 0.5], RWModel=varRWModel)

    gyroBufferData = messaging.AccDataMsgPayload()
    gyroInMsg = messaging.AccDataMsg()
    gyroInMsg.write(gyroBufferData, 0)

    config0 = {
        "vcMsg": vcMsg,
        "rwStateEffector": rwStateEffector,
        "inertialAttFilterRwParamMsg": rwFactory_0.getConfigMessage(),
        "gyroInMsg": gyroInMsg,
        "st_cov": st_cov,
    }
    configure_inertialattfilter(inertialAttFilter0, config0, attitude_measurement_msg)

    inertialAttFilter0Log = inertialAttFilter0.logger(
        ["covar", "state", "cov_S", "innovation"],
        unitTestSupport.samplingTime(simulationTime, simulationTimeStep, 100)
    )
    scSim.AddModelToTask(simTaskName, inertialAttFilter0Log)

    # --- Setup logs ---
    snAttLog, rwLogs = setup_logging(
        scSim, simTaskName,
        unitTestSupport.samplingTime(simulationTime, simulationTimeStep, 100),
        rwMotorTorqueObj, attError, sNavObject, rwStateEffector, numRW
    )

    # --- Simulation duration and timestamps ---
    simTimeSec = duration
    dt_sample = 1.0 / sample_rate_hz
    timeSpan = np.arange(0, simTimeSec + dt_sample, dt_sample)

    scSim.InitializeSimulation()

    # --- Pre-allocate logs ---
    att_log, body_rate_log, rw_omega_log, rw_torque_log = [], [], [], []

    # --- Simulation loop ---
    for t in timeSpan[1:]:
        scSim.ConfigureStopTime(macros.sec2nano(t))
        scSim.ExecuteSimulation()

        # Write noisy star tracker measurement
        if snAttLog.sigma_BN.shape[0] > 0:
            true_att = snAttLog.sigma_BN[-1, :]
            noisy_att = true_att + np.random.normal(0, np.sqrt(st_cov), 3)
            st_1_data.valid = True
            st_1_data.timeTag = int(t * 1E9)
            st_1_data.MRP_BdyInrtl = noisy_att
            attitude_measurement_msg.write(st_1_data, int(t * 1E9))

        # Log states
        att_log.append(snAttLog.sigma_BN[-1, :])
        body_rate_log.append(snAttLog.omega_BN_B[-1, :])
        rw_omega_log.append([rw.Omega for rw in rwFactory.rwList.values()])
        rw_torque_log.append([
            getattr(rw, "u_current", getattr(rw, "motorTorque", np.nan))
            for rw in rwFactory.rwList.values()
        ])

    # --- Package results ---
    timestamps = timeSpan[1:]
    fault_labels = np.ones((len(timestamps), numRW), dtype=np.uint8)  # all 1s since fault is active

    # Last spacecraft state
    rN = np.array(scObject.hub.r_CN_NInit).flatten()
    vN = np.array(scObject.hub.v_CN_NInit).flatten()
    mu = simIncludeGravBody.gravBodyFactory().createEarth().mu
    oe = orbitalMotion.rv2elem_parab(mu, rN, vN)
    sigma = snAttLog.sigma_BN[-1, :]
    omega = snAttLog.omega_BN_B[-1, :]

    last_state_faulty = {
        "oe": oe,
        "rN": rN,
        "vN": vN,
        "sigma": sigma,
        "omega": omega
    }

    dataset_faulty = {
        "timestamps": timestamps,
        "att_log": np.array(att_log),
        "body_rate_log": np.array(body_rate_log),
        "rw_omega_log": np.array(rw_omega_log),
        "rw_torque_log": np.array(rw_torque_log),
        "fault_labels": fault_labels
    }

    return last_state_faulty, dataset_faulty


if __name__ == "__main__":

    sample_rate_hz = 1  # 1 Hz

    # --- Define sequence of runs: (type, duration_sec) ---
    run_sequence = [
        ("healthy", 60),
        ("faulty", 360),
        ("healthy", 60),
        ("faulty", 360),
        ("healthy", 60),
        ("faulty", 360),
        ("healthy", 2340)
    ]

    all_timestamps = []
    all_att_log = []
    all_body_rate_log = []
    all_rw_omega_log = []
    all_rw_torque_log = []
    all_fault_labels = []

    t_offset = 0
    last_state = None  # will store last state of previous run

    for run_type, duration_sec in run_sequence:
        if run_type == "healthy":
            if last_state is None:
                # first healthy run starts fresh
                last_state, dataset = run_healthy_rw_dataset(
                    sample_rate_hz=sample_rate_hz,
                    duration=duration_sec,
                    seed=42
                )
            else:
                # subsequent healthy runs continue from last state
                last_state, dataset = run_healthy_rw_dataset(
                    sample_rate_hz=sample_rate_hz,
                    duration=duration_sec,
                    seed=42,
                    last_state=last_state   # <-- actually passed!
                )
        elif run_type == "faulty":
            # faulty runs always start from last state
            last_state, dataset = run_faulty_rw_dataset(
                last_state_healthy=last_state,
                sample_rate_hz=sample_rate_hz,
                duration=duration_sec,
                seed=42
            )
        else:
            raise ValueError(f"Unknown run type: {run_type}")

        # Append dataset with shifted timestamps
        all_timestamps.append(dataset["timestamps"] + t_offset)
        all_att_log.append(dataset["att_log"])
        all_body_rate_log.append(dataset["body_rate_log"])
        all_rw_omega_log.append(dataset["rw_omega_log"])
        all_rw_torque_log.append(dataset["rw_torque_log"])
        all_fault_labels.append(dataset["fault_labels"])

        t_offset += duration_sec

    # --- Concatenate all arrays ---
    timestamps = np.concatenate(all_timestamps)
    att_log = np.concatenate(all_att_log)
    body_rate_log = np.concatenate(all_body_rate_log)
    rw_omega_log = np.concatenate(all_rw_omega_log)
    rw_torque_log = np.concatenate(all_rw_torque_log)
    fault_labels = np.concatenate(all_fault_labels)

    # --- Save HDF5 ---
    sensors = {
        "attitude_mrp": att_log,
        "body_rates_radps": body_rate_log,
        "rw_wheel_speed_radps": rw_omega_log,
        "rw_motor_torque_Nm": rw_torque_log
    }

    os.makedirs(".", exist_ok=True)
    with h5py.File("faulty_rw_run.h5", "w") as f:
        f.attrs["description"] = "RW dataset: each run continues from last state of previous run"
        f.attrs["sample_rate_hz"] = sample_rate_hz
        f.attrs["duration_sec"] = timestamps[-1]
        f.create_dataset("time_s", data=timestamps)
        f.create_dataset("labels", data=fault_labels)

        g = f.create_group("sensors")
        for k, v in sensors.items():
            g.create_dataset(k, data=v)

    print("[OK] Saved dataset to 'faulty_rw_continuous_sequence.h5'")
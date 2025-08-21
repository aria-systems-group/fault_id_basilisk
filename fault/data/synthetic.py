import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import h5py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib.pyplot as plt  # kept for compatibility with existing imports
from Basilisk import __path__
from Basilisk.architecture import messaging
from Basilisk.utilities import macros, unitTestSupport, simIncludeRW
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
    out_path="healthy_rw_run.h5",
    sample_rate_hz=1,
    duration_hours=12,
    seed: int = 0
):
    """
    Generates a reaction-wheels HEALTHY dataset (no faults) at 1 Hz for 12 hours in HDF5.
    Synchronized timestamps + N-dim fault-label vector (N=3 for RW1/RW2/RW3), all zeros.
    """
    # --- Set seed for reproducibility ---
    np.random.seed(42)  

    # Setup spacecraft and simulation module
    (scSim, scObject,simTaskName, simTimeSec, simTimeStepSec, simulationTime, simulationTimeStep,
        varRWModel, rwFactory, rwStateEffector, numRW, I) = setup_spacecraft_sim(true_mode=0)

    # Setup navigation module
    sNavObject, inertial3DObj, attError, mrpControl = setup_navigation_and_control(scSim, simTaskName)

    # Connect messages
    vcMsg, attitude_measurement_msg, st_cov, rwMotorTorqueObj, st_1_data \
        = setup_messages(scSim, simTaskName, I, rwFactory, scObject, sNavObject, attError, inertial3DObj, mrpControl, rwStateEffector)
    fswRwParamMsg = rwFactory.getConfigMessage()
    mrpControl.rwParamsInMsg.subscribeTo(fswRwParamMsg)
    rwMotorTorqueObj.rwParamsInMsg.subscribeTo(fswRwParamMsg)

    # --- Create multiple inertialUKF (state = MRP, angular_rate) ---
    # create an empty gyro measurement
    gyroBufferData = messaging.AccDataMsgPayload()
    gyroInMsg = messaging.AccDataMsg()
    gyroInMsg.write(gyroBufferData, 0)

    numDataPoints = 100
    samplingTime = unitTestSupport.samplingTime(simulationTime, simulationTimeStep, numDataPoints)

    # 0: nominal filter
    inertialAttFilter0 = inertialUKF.inertialUKF()
    scSim.AddModelToTask(simTaskName, inertialAttFilter0)
    rwFactory_0 = simIncludeRW.rwFactory()
    # create each RW by specifying the RW type, the spin axis gsHat, plus optional arguments
    rwFactory_0.create('Honeywell_HR16', [1, 0, 0], maxMomentum=50., Omega=100.  # RPM
                           , RWModel=varRWModel, 
                           )
    rwFactory_0.create('Honeywell_HR16', [0, 1, 0], maxMomentum=50., Omega=200.  # RPM
                           , RWModel=varRWModel
                           )
    rwFactory_0.create('Honeywell_HR16', [0, 0, 1], maxMomentum=50., Omega=300.  # RPM
                           , rWB_B=[0.5, 0.5, 0.5]  # meters
                           , RWModel=varRWModel,
                           )
    config0 = {
        "vcMsg": vcMsg,
        "rwStateEffector": rwStateEffector, 
        "inertialAttFilterRwParamMsg": rwFactory_0.getConfigMessage(), 
        "gyroInMsg": gyroInMsg,
        "st_cov": st_cov,
    }
    configure_inertialattfilter(inertialAttFilter0, config0, attitude_measurement_msg)
    inertialAttFilter0Log = inertialAttFilter0.logger(["covar", "state", "cov_S", "innovation"], samplingTime)
    scSim.AddModelToTask(simTaskName, inertialAttFilter0Log)

     # Setup logs
    snAttLog, rwLogs = setup_logging(scSim, simTaskName, samplingTime, rwMotorTorqueObj, 
                                     attError, sNavObject, rwStateEffector, numRW)

    # --- Simulation duration and sampling ---
    # simTimeSec = duration_hours * 3600.0
    simTimeSec = duration_hours * 3600
    dt_sample = 1.0 / sample_rate_hz
    timeSpan = np.arange(0, simTimeSec + dt_sample, dt_sample)

    # --- Initialize Simulation ---
    scSim.InitializeSimulation()

    # Pre-allocate logs
    sensors = {}

    rw_omega_log = []
    rw_torque_log = []
    att_log = []
    body_rate_log = []

    # --- Run Simulation with 1Hz samples ---
    for t in timeSpan[1:]:
        scSim.ConfigureStopTime(macros.sec2nano(t))
        scSim.ExecuteSimulation()

        # obtain true star tracker measurement
        if snAttLog.sigma_BN.shape[0] > 0:
            true_att = snAttLog.sigma_BN[-1, :]
            true_att_with_noise = true_att + np.random.normal(0, np.sqrt(st_cov), 3)
            st_1_data.valid = True
            st_1_data.timeTag = int(t * 1E9)
            st_1_data.MRP_BdyInrtl = true_att_with_noise
            attitude_measurement_msg.write(st_1_data, int(t * 1E9))
        
         # --- Log true attitude and body rates manually ---
        att_snapshot = snAttLog.sigma_BN[-1, :]          # MRP vector
        rate_snapshot = snAttLog.omega_BN_B[-1, :]       # body rates
        att_log.append(att_snapshot)
        body_rate_log.append(rate_snapshot)

        # --- Log RW Omega directly at this timestep ---
        rw_omega_snapshot = [rw.Omega for rw in rwFactory.rwList.values()]
        rw_omega_log.append(rw_omega_snapshot)

        # --- Log RW motor torque at this timestep ---
        rw_torque_snapshot = [
            getattr(rw, "u_current", getattr(rw, "motorTorque", np.nan))
            for rw in rwFactory.rwList.values()
        ]
        rw_torque_log.append(rw_torque_snapshot)

    # --- Collect data for HDF5 ---
    timestamps = timeSpan[1:]  # seconds, aligned with sample rate

    # Fault label vector: all zeros for healthy run
    fault_labels = np.zeros((len(timestamps), numRW), dtype=np.uint8)

    # --- After the loop, convert to numpy arrays ---
    sensors["attitude_mrp"] = np.array(att_log)
    sensors["body_rates_radps"] = np.array(body_rate_log)
    sensors["rw_wheel_speed_radps"] = np.array(rw_omega_log)      # shape: (timesteps, numRW)
    sensors["rw_motor_torque_Nm"] = np.array(rw_torque_log)       # shape: (timesteps, numRW)

    # --- Save to HDF5 ---
    os.makedirs(os.path.dirname("healthy_rw_run.h5") or ".", exist_ok=True)
    with h5py.File("healthy_rw_run.h5", "w") as f:
        f.attrs["description"] = "Healthy reaction wheel dataset (no faults)"
        f.attrs["sample_rate_hz"] = sample_rate_hz
        f.attrs["duration_hours"] = duration_hours
        f.create_dataset("time_s", data=timestamps)
        f.create_dataset("labels", data=fault_labels)
        
        g = f.create_group("sensors")
        for k, v in sensors.items():
            g.create_dataset(k, data=v)

    print(f"[OK] Saved healthy dataset to 'healthy_rw_run.h5'")



if __name__ == "__main__":
    run_healthy_rw_dataset(
        out_path="healthy_rw_run.h5",
        sample_rate_hz=1,
        duration_hours=1,
        seed=42
    )
"""
Test if one can use many native forward simulation of BSK to construct UKF.
NOTE: I am not quite sure ...
    - If the UKF simulations do not use the feedback law as the true S/C --> the UKF still works, which is not expected.
    - There is something wrong when combining the forward simulation into UKF estimates.
"""

import os
import pytest
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from scipy.stats import chi2

# The path to the location of Basilisk
# Used to get the location of supporting data.
from Basilisk import __path__
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import (mrpFeedback, attTrackingError,
                                    inertial3D, rwMotorTorque)
from Basilisk.simulation import reactionWheelStateEffector, simpleNav, spacecraft
from Basilisk.utilities import (SimulationBaseClass, macros,
                                orbitalMotion, simIncludeGravBody,
                                simIncludeRW, unitTestSupport, vizSupport)

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])


def plot_filter_result_sigma(timeData, state, state_est, cov_est, state_meas):
    timeData = timeData/(1E+9)/(60.0)
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    
    for idx, ax in enumerate(axs):
        color = unitTestSupport.getLineColor(idx, 3)
        
        # True state
        ax.plot(timeData, state[idx, :],
                color=color,
                label=rf'$x_{{{idx+1}}}$')
        
        # Estimated state
        ax.plot(timeData, state_est[idx, :],
                color=color,
                linestyle='--',
                label=rf'$\hat{{x}}_{{{idx+1}}}$')
        
        # Measured state
        ax.scatter(timeData, state_meas[idx, :],
                color="black",
        )
        
        # ±6 std‐dev band
        std5  = 6 * np.sqrt(cov_est[idx, :])
        upper = state_est[idx, :] + std5
        lower = state_est[idx, :] - std5
        ax.fill_between(timeData, lower, upper,
                        color=color,
                        alpha=0.3,
                        label=r'$\pm6\sigma$')
        
        ax.set_ylabel(rf'$x_{{{idx+1}}}$')      # y-label per subplot
        ax.legend(loc='upper right', fontsize='small')
        # margin = 0.0
        # ax.set_ylim([state[:, idx].min()-margin, state[:, idx].max()+margin])
    # Common x‑label on the bottom subplot
    axs[-1].set_xlabel('Time [min]')


def plot_filter_result_omega(timeData, state, state_est, cov_est):
    timeData = timeData/(1E+9)/(60.0)
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    
    for idx, ax in enumerate(axs):
        color = unitTestSupport.getLineColor(idx, 3)
        
        # True state
        ax.plot(timeData, state[idx, :],
                color=color,
                label=rf'$x_{{{idx+4}}}$')
        
        # Estimated state
        ax.plot(timeData, state_est[idx, :],
                color=color,
                linestyle='--',
                label=rf'$\hat{{x}}_{{{idx+4}}}$')
        
        # ±6 std‐dev band
        std5  = 6 * np.sqrt(cov_est[idx, :])
        upper = state_est[idx, :] + std5
        lower = state_est[idx, :] - std5
        ax.fill_between(timeData, lower, upper,
                        color=color,
                        alpha=0.3,
                        label=r'$\pm6\sigma$')
        
        ax.set_ylabel(rf'$x_{{{idx+1}}}$')      # y-label per subplot
        ax.legend(loc='upper right', fontsize='small')
        # margin = 0.0
        # ax.set_ylim([state[:, idx].min()-margin, state[:, idx].max()+margin])
    # Common x‑label on the bottom subplot
    axs[-1].set_xlabel('Time [min]')


# def plot_attitude_error(timeData, dataSigmaBR):
#     """Plot the attitude errors."""
#     plt.figure(1)
#     for idx in range(3):
#         plt.plot(timeData, dataSigmaBR[:, idx],
#                  color=unitTestSupport.getLineColor(idx, 3),
#                  label=r'$\sigma_' + str(idx) + '$')
#     plt.legend(loc='lower right')
#     plt.xlabel('Time [min]')
#     plt.ylabel(r'Attitude Error $\sigma_{B/R}$')

# def plot_filter_result_sigma(timeData, state, state_est, cov_est):
#     timeData = timeData[1:]
#     fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    
#     for idx, ax in enumerate(axs):
#         color = unitTestSupport.getLineColor(idx, 3)
        
#         # True state
#         ax.plot(timeData, state[1:, idx],
#                 color=color,
#                 label=rf'$x_{{{idx+1}}}$')
        
#         # Estimated state
#         ax.plot(timeData, state_est[1:, idx],
#                 color=color,
#                 linestyle='--',
#                 label=rf'$\hat{{x}}_{{{idx+1}}}$')
        
#         # ±6 std‐dev band
#         std5  = 6 * np.sqrt(cov_est[1:, idx])
#         upper = state_est[1:, idx] + std5
#         lower = state_est[1:, idx] - std5
#         ax.fill_between(timeData, lower, upper,
#                         color=color,
#                         alpha=0.3,
#                         label=r'$\pm6\sigma$')
        
#         ax.set_ylabel(rf'$x_{{{idx+1}}}$')      # y-label per subplot
#         ax.legend(loc='upper right', fontsize='small')
#         # margin = 0.0
#         # ax.set_ylim([state[:, idx].min()-margin, state[:, idx].max()+margin])
#     # Common x‑label on the bottom subplot
#     axs[-1].set_xlabel('Time [min]')

# def plot_filter_result_omega(timeData, state, state_est, cov_est):
#     timeData = timeData[1:]
#     fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    
#     for idx, ax in enumerate(axs):
#         color = unitTestSupport.getLineColor(idx, 3)
        
#         # True state
#         ax.plot(timeData, state[1:, idx],
#                 color=color,
#                 label=rf'$x_{{{idx+3}}}$')
        
#         # Estimated state
#         ax.plot(timeData, state_est[1:, idx],
#                 color=color,
#                 linestyle='--',
#                 label=rf'$\hat{{x}}_{{{idx+3}}}$')
        
#         # ±5 std‐dev band
#         std5  = 6 * np.sqrt(cov_est[1:, idx])
#         upper = state_est[1:, idx] + std5
#         lower = state_est[1:, idx] - std5
#         ax.fill_between(timeData, lower, upper,
#                         color=color,
#                         alpha=0.3,
#                         label=r'$\pm6\sigma$')
        
#         ax.set_ylabel(rf'$x_{{{idx+3}}}$')      # y-label per subplot
#         ax.legend(loc='upper right', fontsize='small')
#         # ax.set_ylim([state[:, idx].min(), state[:, idx].max()])
#     # Common x‑label on the bottom subplot
#     axs[-1].set_xlabel('Time [min]')

# def plot_filter_chisquare(dataChiSquare):
#     p = 0.05      # confidence level
#     dof = 3       # degrees of freedom
#     chi_ub = chi2.ppf(1-0.5*p, dof)
#     chi_lb = chi2.ppf(0.5*p, dof)
#     plt.figure()
#     plt.scatter(np.arange(len(dataChiSquare)), dataChiSquare, color="black", s=10, label=r"$\chi^2$")
#     plt.axhline(y=chi_ub, color='r', linestyle='--', label=r"$\chi^2$ upper threshold")
#     plt.axhline(y=chi_lb, color='b', linestyle='--', label=r"$\chi^2$ lower threshold")
#     plt.legend(loc='upper right')

# def plot_rw_cmd_torque(timeData, dataUsReq, numRW):
#     """Plot the RW command torques."""
#     plt.figure(2)
#     for idx in range(3):
#         plt.plot(timeData, dataUsReq[:, idx],
#                  '--',
#                  color=unitTestSupport.getLineColor(idx, numRW),
#                  label=r'$\hat u_{s,' + str(idx) + '}$')
#     plt.legend(loc='lower right')
#     plt.xlabel('Time [min]')
#     plt.ylabel('RW Motor Torque (Nm)')

# def plot_rw_motor_torque(timeData, dataUsReq, dataRW, numRW):
#     """Plot the RW actual motor torques."""
#     plt.figure(2)
#     for idx in range(3):
#         plt.plot(timeData, dataUsReq[:, idx],
#                  '--',
#                  color=unitTestSupport.getLineColor(idx, numRW),
#                  label=r'$\hat u_{s,' + str(idx) + '}$')
#         plt.plot(timeData, dataRW[idx],
#                  color=unitTestSupport.getLineColor(idx, numRW),
#                  label='$u_{s,' + str(idx) + '}$')
#     plt.legend(loc='lower right')
#     plt.xlabel('Time [min]')
#     plt.ylabel('RW Motor Torque (Nm)')

# def plot_rate_error(timeData, dataOmegaBR):
#     """Plot the body angular velocity rate tracking errors."""
#     plt.figure(3)
#     for idx in range(3):
#         plt.plot(timeData, dataOmegaBR[:, idx],
#                  color=unitTestSupport.getLineColor(idx, 3),
#                  label=r'$\omega_{BR,' + str(idx) + '}$')
#     plt.legend(loc='lower right')
#     plt.xlabel('Time [min]')
#     plt.ylabel('Rate Tracking Error (rad/s) ')

# def plot_rw_speeds(timeData, dataOmegaRW, numRW):
#     """Plot the RW spin rates."""
#     plt.figure(4)
#     for idx in range(numRW):
#         plt.plot(timeData, dataOmegaRW[:, idx] / macros.RPM,
#                  color=unitTestSupport.getLineColor(idx, numRW),
#                  label=r'$\Omega_{' + str(idx) + '}$')
#     plt.legend(loc='lower right')
#     plt.xlabel('Time [min]')
#     plt.ylabel('RW Speed (RPM) ')

# def test_two_sc(show_plots=False):
#     """
#     Runs a 600-s sim of two independent spacecraft,
#     each with its own nav → guidance → att-error → MRP control chain,
#     and three reaction wheels.  No UKF is used.
#     """
#     # --- Simulation setup ---
#     sim = SimulationBaseClass.SimBaseClass()
#     dynProcess = sim.CreateNewProcess("simProcess")
#     simTimeSec    = 600.0
#     simTimeNano   = macros.sec2nano(simTimeSec)
#     simDtSec      = 0.1
#     simDtNano     = macros.sec2nano(simDtSec)
#     dynProcess.addTask(sim.CreateNewTask("simTask", simDtNano))

#     # --- Central gravity (Earth) ---
#     gravFactory = simIncludeGravBody.gravBodyFactory()
#     earth = gravFactory.createEarth()
#     earth.isCentralBody = True
#     mu = earth.mu

#     # --- Initial orbit state ---
#     oe = orbitalMotion.ClassicElements()
#     oe.a     = 1.0e7
#     oe.e     = 0.01
#     oe.i     = 33.3 * macros.D2R
#     oe.Omega = 48.2 * macros.D2R
#     oe.omega = 347.8 * macros.D2R
#     oe.f     = 85.3 * macros.D2R
#     rN, vN = orbitalMotion.elem2rv(mu, oe)

#     # --- Spacecraft inertias (same for both) ---
#     Ivals = [900., 0., 0.,
#              0., 800., 0.,
#              0.,   0., 600.]

#     # --- Create two spacecraft bodies ---
#     scList = []
#     for tag in ("bsk-Sat0", "bsk-Sat1"):
#         sc = spacecraft.Spacecraft()
#         sc.ModelTag = tag

#         sc.hub.mHub           = 750.0
#         sc.hub.r_BcB_B        = [[0.0],[0.0],[0.0]]
#         sc.hub.IHubPntBc_B    = unitTestSupport.np2EigenMatrix3d(Ivals)
#         sc.hub.r_CN_NInit     = rN
#         sc.hub.v_CN_NInit     = vN
#         sc.hub.sigma_BNInit   = [[0.1],[0.2],[-0.3]]
#         sc.hub.omega_BN_BInit = [[0.0],[0.0],[0.0]]

#         sim.AddModelToTask("simTask", sc, 1)
#         gravFactory.addBodiesTo(sc)

#         scList.append(sc)

#     # --- Reaction wheels (shared factory) ---
#     rwFactory = simIncludeRW.rwFactory()
#     model = messaging.BalancedWheels
#     rwFactory.create('Honeywell_HR16', [1,0,0], maxMomentum=50., Omega=100., RWModel=model)
#     rwFactory.create('Honeywell_HR16', [0,1,0], maxMomentum=50., Omega=200., RWModel=model)
#     rwFactory.create('Honeywell_HR16', [0,0,1], maxMomentum=50., Omega=300., rWB_B=[0.5,0.5,0.5], RWModel=model)
#     numRW = rwFactory.getNumOfDevices()

#     # --- Single shared VehicleConfig message ---
#     vcPayload = messaging.VehicleConfigMsgPayload()
#     vcPayload.ISCPntB_B = Ivals
#     vcMsg = messaging.VehicleConfigMsg().write(vcPayload)

#     # --- Simulation logging cadence ---
#     samplingTime = unitTestSupport.samplingTime(simTimeNano, simDtNano, 100)

#     # --- Build each spacecraft’s FSW chain and logging ---
#     for idx, sc in enumerate(scList):
#         # 1) Reaction‐wheel effector
#         rwEff = reactionWheelStateEffector.ReactionWheelStateEffector()
#         rwEff.ModelTag = f"RW_Effector_{idx}"
#         rwFactory.addToSpacecraft(sc.ModelTag, rwEff, sc)
#         sim.AddModelToTask("simTask", rwEff, 2)

#         # 2) SimpleNav
#         nav = simpleNav.SimpleNav()
#         nav.ModelTag = f"SimpleNav_{idx}"
#         sim.AddModelToTask("simTask", nav)
#         nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)

#         # 3) Inertial guidance
#         guid = inertial3D.inertial3D()
#         guid.ModelTag = f"Inertial3D_{idx}"
#         guid.sigma_R0N = [0., 0., 0.]
#         sim.AddModelToTask("simTask", guid)

#         # 4) Attitude‐error
#         err = attTrackingError.attTrackingError()
#         err.ModelTag = f"AttError_{idx}"
#         sim.AddModelToTask("simTask", err)
#         err.attNavInMsg.subscribeTo(nav.attOutMsg)
#         err.attRefInMsg.subscribeTo(guid.attRefOutMsg)

#         # 5) MRP feedback controller
#         ctrl = mrpFeedback.mrpFeedback()
#         ctrl.ModelTag = f"MRP_Ctrl_{idx}"
#         sim.AddModelToTask("simTask", ctrl)
#         ctrl.guidInMsg.subscribeTo(err.attGuidOutMsg)

#         # — subscribe the single vehicle config message —
#         ctrl.vehConfigInMsg.subscribeTo(vcMsg)

#         # 6) Reaction‐wheel config message
#         rwCfgMsg = rwFactory.getConfigMessage()
#         ctrl.rwParamsInMsg.subscribeTo(rwCfgMsg)
#         ctrl.rwSpeedsInMsg .subscribeTo(rwEff.rwSpeedOutMsg)

#         # 7) Controller gains
#         ctrl.K  =  3.5
#         ctrl.Ki = -1.0    # disable integrator
#         ctrl.P  = 30.0
#         ctrl.integralLimit = 2.0 / ctrl.Ki * 0.1

#         # 8) RW motor torque actuator
#         rwMotor = rwMotorTorque.rwMotorTorque()
#         rwMotor.ModelTag = f"RW_Motor_{idx}"
#         sim.AddModelToTask("simTask", rwMotor)
#         rwMotor.controlAxes_B   = [1,0,0, 0,1,0, 0,0,1]
#         rwMotor.rwParamsInMsg   .subscribeTo(rwCfgMsg)
#         rwMotor.vehControlInMsg .subscribeTo(ctrl.cmdTorqueOutMsg)

#         # 9) Data logging
#         logs = [
#             sc.scStateOutMsg .recorder(samplingTime),
#             nav.transOutMsg   .recorder(samplingTime),
#             err.attGuidOutMsg .recorder(samplingTime),
#             rwEff.rwSpeedOutMsg.recorder(samplingTime)
#         ]
#         for lg in logs:
#             sim.AddModelToTask("simTask", lg)

#         # 10) Optional Unity viz
#         if show_plots:
#             vizSupport.enableUnityVisualization(
#                 sim, "simTask", sc,
#                 rwEffectorList=rwEff,
#                 saveFile=f"two_sc_no_ukf_{idx}.bsk"
#             )

#     # --- Run the sim ---
#     sim.InitializeSimulation()
#     timeSpan = np.arange(0.0, simTimeSec + simDtSec, simDtSec)
#     for t in timeSpan[1:]:
#         sim.ConfigureStopTime(macros.sec2nano(t))
#         sim.ExecuteSimulation()


from numpy.linalg import cholesky


def generate_sigma_points(x, P, alpha=1e-3, beta=2.0, kappa=0.0):
    """
    Generate 2n+1 sigma points (Unscented Transform) for state x (n×1) and covariance P (n×n).
    Returns:
        sigma_pts: array[(2n+1)×n] of sigma points
        Wm, Wc:   weight vectors for mean and covariance (length 2n+1)
    """
    n   = x.size
    lam = alpha**2 * (n + kappa) - n
    sqrt_mat = cholesky((n + lam) * P)
    # weights
    Wm = np.full(2*n+1, 1.0 / (2*(n + lam)))
    Wc = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)
    # pack sigma points
    sigma_pts = np.zeros((2*n+1, n))
    sigma_pts[0] = x.copy()
    for i in range(n):
        sigma_pts[i+1]   = x + sqrt_mat[:, i]
        sigma_pts[n+1+i] = x - sqrt_mat[:, i]
    return sigma_pts, Wm, Wc


def set_navigation():
    # --- choose your 1-sigma values ---
    posSigma  = 5.0     # [m]
    velSigma  = 0.035      # [m/s]
    attSigma  = 1e-4    # [rad]
    rateSigma = 1e-4    # [rad/s]
    sunSigma  = 0.0      # [rad] (example)
    dvSigma   = 0.0     # [m/s] (example)

    # --- build the PMatrix (Cholesky of covariance) ---
    pMatrix = np.zeros((18,18))
    # position errors (indices 0–2)
    pMatrix[0,0] = posSigma
    pMatrix[1,1] = posSigma
    pMatrix[2,2] = posSigma
    # velocity errors (3–5)
    pMatrix[3,3] = velSigma
    pMatrix[4,4] = velSigma
    pMatrix[5,5] = velSigma
    # attitude errors (6–8)
    pMatrix[6,6] = attSigma
    pMatrix[7,7] = attSigma
    pMatrix[8,8] = attSigma
    # body-rate errors (9–11)
    pMatrix[9,9]   = rateSigma
    pMatrix[10,10] = rateSigma
    pMatrix[11,11] = rateSigma
    # sun-point errors (12–14)
    pMatrix[12,12] = sunSigma
    pMatrix[13,13] = sunSigma
    pMatrix[14,14] = sunSigma
    # accumulated ΔV errors (15–17)
    pMatrix[15,15] = dvSigma
    pMatrix[16,16] = dvSigma
    pMatrix[17,17] = dvSigma
    return pMatrix


def test_sigma_point_forward_propagation():
    """
    Demonstrate UT-prediction: spawn clones for each sigma point,
    propagate them one step, then reconstruct predicted mean & covariance.
    """
    np.random.seed(42)

    # Simulation parameters
    dt_sec      = 1.0
    total_time  = 400.0            
    steps       = int(total_time / dt_sec)

    # --- Build sim / process / task ---
    sim   = SimulationBaseClass.SimBaseClass()
    proc  = sim.CreateNewProcess("dynProcess")
    task  = sim.CreateNewTask("task", macros.sec2nano(dt_sec))
    proc.addTask(task)

    # --- 2) Add gravity (Earth) ---
    gravF = simIncludeGravBody.gravBodyFactory()
    earth = gravF.createEarth()
    earth.isCentralBody = True
    mu = earth.mu

    # Initial orbit via classical elements
    oe = orbitalMotion.ClassicElements()
    oe.a = 10000000.0  # meters
    oe.e = 0.01
    oe.i = 33.3 * macros.D2R
    oe.Omega = 48.2 * macros.D2R
    oe.omega = 347.8 * macros.D2R
    oe.f = 85.3 * macros.D2R
    r0, v0 = orbitalMotion.elem2rv(mu, oe)

    # --- True Spacecraft ---
    x0 = np.hstack((np.array([0.1, 0.2, -0.3]), np.zeros(3)))
    P0 = np.eye(6) * 1e-2
    sc_true = spacecraft.Spacecraft()
    sc_true.ModelTag = "sc-true"
    sc_true.hub.mHub           = 750.0
    sc_true.hub.r_BcB_B        = [[0.0],[0.0],[0.0]]
    Ivals = [900.,0.,0., 0.,800.,0., 0.,0.,600.]
    sc_true.hub.IHubPntBc_B    = unitTestSupport.np2EigenMatrix3d(Ivals)
    sc_true.hub.r_CN_NInit     = r0
    sc_true.hub.v_CN_NInit     = v0
    # set attitude+rate from sigma point idx
    sc_true.hub.sigma_BNInit   = x0[0:3]
    sc_true.hub.omega_BN_BInit = x0[3:6]
    # add to sim & gravity
    sim.AddModelToTask("task", sc_true, 1)
    gravF.addBodiesTo(sc_true)
    scTrueLog = sc_true.scStateOutMsg.recorder()
    sim.AddModelToTask("task", scTrueLog)

    # --- Setup Reaction Wheels ---
    rwFactory = simIncludeRW.rwFactory()
    varRWModel = messaging.BalancedWheels
    RW1 = rwFactory.create('Honeywell_HR16', [1, 0, 0], maxMomentum=100., Omega=100.  # RPM
                           , RWModel=varRWModel
                           , Omega_max=1000.0
                           )
    RW2 = rwFactory.create('Honeywell_HR16', [0, 1, 0], maxMomentum=100., Omega=200.  # RPM
                           , RWModel=varRWModel
                           )
    rwFactory.create('Honeywell_HR16', [0, 0, 1], maxMomentum=100., Omega=300.  # RPM
                           , rWB_B=[0.5, 0.5, 0.5]  # meters
                           , RWModel=varRWModel
                           )
    RW1.Omega_max = RW2.Omega_max
    numRW = rwFactory.getNumOfDevices()
    # --- Connect Reaction Wheels to Spacecraft via rwStateEffector ---
    rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwStateEffector.ModelTag = "RW_cluster"
    rwFactory.addToSpacecraft(sc_true.ModelTag, rwStateEffector, sc_true)
    # add RW object array to the simulation process.  This is required for the UpdateState() method
    # to be called which logs the RW states
    sim.AddModelToTask("task", rwStateEffector, 2)

    # --- Setup Navigation --- 
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"
    # sNavObject.PMatrix = set_navigation()
    # sNavObject.crossAtt = True
    sim.AddModelToTask("task", sNavObject)
    sNavAttLog = sNavObject.attOutMsg.recorder()
    sim.AddModelToTask("task", sNavAttLog)
    # setup inertial3D guidance module
    inertial3DObj = inertial3D.inertial3D()
    inertial3DObj.ModelTag = "inertial3D"
    sim.AddModelToTask("task", inertial3DObj)
    inertial3DObj.sigma_R0N = [0., 0., 0.]  # set the desired inertial orientation
    # setup the attitude tracking error evaluation module
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    sim.AddModelToTask("task", attError)
    # setup the MRP Feedback control module
    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedback"
    sim.AddModelToTask("task", mrpControl)
    mrpControl.K = 3.5
    mrpControl.Ki = -1  # make value negative to turn off integral feedback
    mrpControl.P = 30.0
    mrpControl.integralLimit = 2. / mrpControl.Ki * 0.1

    # --- Connect messages ---
    # create the FSW vehicle configuration message
    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    vehicleConfigOut.ISCPntB_B = Ivals  # use the same inertia in the FSW algorithm as in the simulation
    vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)
    # create the FSW reaction wheel configuration message
    fswRwParamMsg = rwFactory.getConfigMessage()
    # connect navigation to spacecraft
    sNavObject.scStateInMsg.subscribeTo(sc_true.scStateOutMsg)
    # connect att control error
    attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
    attError.attRefInMsg.subscribeTo(inertial3DObj.attRefOutMsg)
    # connect mrp control law
    mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
    mrpControl.vehConfigInMsg.subscribeTo(vcMsg)
    mrpControl.rwParamsInMsg.subscribeTo(fswRwParamMsg)
    mrpControl.rwSpeedsInMsg.subscribeTo(rwStateEffector.rwSpeedOutMsg)
    # create and connect RW motor torqe
    rwMotorTorqueObj = rwMotorTorque.rwMotorTorque()
    rwMotorTorqueObj.ModelTag = "rwMotorTorque"
    sim.AddModelToTask("task", rwMotorTorqueObj)
    #   make the RW control all three body axes
    controlAxes_B = [
        1, 0, 0, 
        0, 1, 0, 
        0, 0, 1
    ]
    rwMotorTorqueObj.controlAxes_B = controlAxes_B
    rwMotorTorqueObj.rwParamsInMsg.subscribeTo(fswRwParamMsg)
    rwMotorTorqueObj.vehControlInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)
    # connect rwStateEffector to rwMotorTorque
    rwStateEffector.rwMotorCmdInMsg.subscribeTo(rwMotorTorqueObj.rwMotorTorqueOutMsg)

    # --- UKF init ---
    Q     = np.diag([1e-8, 1e-8, 1e-8, 1e-7, 1e-7, 1e-7])
    R     = np.eye(3) * 1e-3  # measurement noise on σ only
    x_est = x0
    P_est = P0
    # spawn initial sigma-point clones
    sigma_pts, Wm, Wc = generate_sigma_points(x_est, P_est)
    n_sigma = sigma_pts.shape[0]
    scList, scLogList = [], []

    # --- RW config --
    rwFactory_UKF = simIncludeRW.rwFactory()
    RW1_UKF = rwFactory_UKF.create('Honeywell_HR16', [1, 0, 0], maxMomentum=100., Omega=100.  # RPM,
                           , RWModel=varRWModel
                           , Omega_max=1000.0
                           )
    RW2_UKF = rwFactory_UKF.create('Honeywell_HR16', [0, 1, 0], maxMomentum=100., Omega=200.  # RPM
                           , RWModel=varRWModel
                           )
    rwFactory_UKF.create('Honeywell_HR16', [0, 0, 1], maxMomentum=100., Omega=300.  # RPM
                           , rWB_B=[0.5, 0.5, 0.5]  # meters
                           , RWModel=varRWModel
                           )
    RW1_UKF.Omega_max = RW2_UKF.Omega_max
    fswRwParamMsg_UKF = rwFactory_UKF.getConfigMessage()
    
    for i in range(n_sigma):
        sc = spacecraft.Spacecraft()
        sc.ModelTag = f"scSigma_{i}"
        sc.hub.mHub           = sc_true.hub.mHub
        sc.hub.r_BcB_B        = sc_true.hub.r_BcB_B
        sc.hub.IHubPntBc_B    = sc_true.hub.IHubPntBc_B
        sc.hub.r_CN_NInit     = r0
        sc.hub.v_CN_NInit     = v0
        sc.hub.sigma_BNInit   = sigma_pts[i,0:3].reshape(3,1)
        sc.hub.omega_BN_BInit = sigma_pts[i,3:6].reshape(3,1)
        sim.AddModelToTask("task", sc, 1)
        gravF.addBodiesTo(sc)

        log = sc.scStateOutMsg.recorder()
        sim.AddModelToTask("task", log)
        scList.append(sc)
        scLogList.append(log)

        # # --- Control Law ---
        # rwStateEffector_UKF = reactionWheelStateEffector.ReactionWheelStateEffector()
        # rwStateEffector_UKF.ModelTag = f"RW_cluster_{i}"
        # rwFactory_UKF.addToSpacecraft(sc.ModelTag, rwStateEffector_UKF, sc)
        # sim.AddModelToTask("task", rwStateEffector_UKF)

        # sNavObject_UKF = simpleNav.SimpleNav()
        # sNavObject_UKF.ModelTag = f"SimpleNavigation_{i}"
        # sim.AddModelToTask("task", sNavObject_UKF)

        # attError_UKF = attTrackingError.attTrackingError()
        # attError_UKF.ModelTag = f"attErrorInertial3D_{i}"
        # sim.AddModelToTask("task", attError_UKF)

        # mrpControl_UKF = mrpFeedback.mrpFeedback()
        # mrpControl_UKF.ModelTag = f"mrpFeedback_{i}"
        # sim.AddModelToTask("task", mrpControl_UKF)
        # mrpControl_UKF.K = mrpControl.K
        # mrpControl_UKF.Ki = mrpControl.Ki
        # mrpControl_UKF.P = mrpControl.P
        # mrpControl_UKF.integralLimit = mrpControl.integralLimit

        # # connect navigation to spacecraft
        # sNavObject_UKF.scStateInMsg.subscribeTo(sc.scStateOutMsg)

        # # connect att control error
        # attError_UKF.attNavInMsg.subscribeTo(sNavObject_UKF.attOutMsg)
        # attError_UKF.attRefInMsg.subscribeTo(inertial3DObj.attRefOutMsg)

        # # connect mrp control law
        # mrpControl_UKF.guidInMsg.subscribeTo(attError_UKF.attGuidOutMsg)
        # mrpControl_UKF.vehConfigInMsg.subscribeTo(vcMsg)
        # mrpControl_UKF.rwParamsInMsg.subscribeTo(fswRwParamMsg_UKF)
        # mrpControl_UKF.rwSpeedsInMsg.subscribeTo(rwStateEffector_UKF.rwSpeedOutMsg)

        # # create and connect RW motor torqe
        # rwMotorTorqueObj_UKF = rwMotorTorque.rwMotorTorque()
        # rwMotorTorqueObj_UKF.ModelTag = f"rwMotorTorque_{i}"
        # rwMotorTorqueObj_UKF.controlAxes_B = controlAxes_B
        # sim.AddModelToTask("task", rwMotorTorqueObj_UKF)
        # rwMotorTorqueObj_UKF.rwParamsInMsg.subscribeTo(fswRwParamMsg_UKF)
        # rwMotorTorqueObj_UKF.vehControlInMsg.subscribeTo(mrpControl_UKF.cmdTorqueOutMsg)

        # # connect rwStateEffector to rwMotorTorque
        # rwStateEffector_UKF.rwMotorCmdInMsg.subscribeTo(rwMotorTorqueObj_UKF.rwMotorTorqueOutMsg)


    # --- Time loop ---
    sim.InitializeSimulation()
    current_time = 0.0
    UKF_result = {
        "timeNanos": [],
        "x_est": [],
        "P_est": [],
        "chi_square": [],
    }
    Measurements = {
        "timeNanos": [],
        "sigma_BN": [],
    }

    for kk in range(steps):
        # propagate true & clones one step
        current_time += dt_sec
        sim.ConfigureStopTime(macros.sec2nano(current_time))
        sim.ExecuteSimulation()

        # --- UKF ---
        # 1) collect propagated sigma-states
        prop = np.zeros_like(sigma_pts)
        for i, log in enumerate(scLogList):
            prop[i, 0:3] = log.sigma_BN[-1, :]
            prop[i, 3:6] = log.omega_BN_B[-1, :]

        # 2) UT-predict mean & covariance
        x_pred = (Wm @ prop)
        P_pred = np.zeros((6,6))
        for i in range(n_sigma):
            d = (prop[i] - x_pred).reshape(6,1)
            P_pred += Wc[i] * (d @ d.T)
        P_pred += Q

        # 3) measurement from true SC
        y_true = sNavAttLog.sigma_BN[-1, :] + np.random.normal(0, np.sqrt(R[0,0]), 3)
        Measurements["timeNanos"].append(macros.sec2nano(current_time))
        Measurements["sigma_BN"].append(y_true)

        # 4) UT-predicted measurement mean & cov
        Y_sigmas = prop[:, 0:3]
        y_pred = (Wm @ Y_sigmas)
        P_xy  = np.zeros((6,3)); P_yy = np.zeros((3,3))
        for i in range(n_sigma):
            dx = (prop[i] - x_pred).reshape(6,1)
            dy = (Y_sigmas[i]    - y_pred).reshape(3,1)
            P_xy += Wc[i] * (dx @ dy.T)
            P_yy += Wc[i] * (dy @ dy.T)
        S = P_yy + R

        # 5) Kalman gain & update
        K     = P_xy @ np.linalg.inv(S)
        x_est = x_pred + K @ (y_true - y_pred)
        P_est = P_pred - K @ S @ K.T

        r = (y_true - y_pred).reshape(-1,1)      # make it a column vector
        chi_square = (r.T @ np.linalg.inv(S) @ r).reshape(-1,)

        UKF_result["timeNanos"].append(macros.sec2nano(current_time))
        UKF_result["x_est"].append(x_est)
        UKF_result["P_est"].append(P_est)
        UKF_result["chi_square"].append(chi_square)

        # 6) regenerate sigma points around posterior
        sigma_pts, Wm, Wc = generate_sigma_points(x_est, P_est)

        # 7) reset clones and Reset
        for i, sc in enumerate(scList):
            sc.hub.sigma_BNInit   = sigma_pts[i,0:3]
            sc.hub.omega_BN_BInit = sigma_pts[i,3:6]
    
    # --- See Simulation Result ---
    dataTrueAtt = np.vstack((scTrueLog.times(), scTrueLog.sigma_BN.T, scTrueLog.omega_BN_B.T))
    dataTrueAtt = dataTrueAtt[:, 1:]
    dataMeasAtt = np.vstack((np.array(Measurements["timeNanos"]),
                             np.array(Measurements["sigma_BN"]).T
                             ))
    # --- assert
    _tmp = np.array(UKF_result["P_est"])
    dataFilterSigmaDiagCov = np.diagonal(_tmp, 
                                         axis1=1, axis2=2)[:, :3].T
    dataFilterOmegaDiagCov = np.diagonal(_tmp, 
                                         axis1=1, axis2=2)[:, 3:].T
    dataUKFAtt = np.vstack((np.array(UKF_result["timeNanos"]), 
                            np.array(UKF_result["x_est"]).T,
                            dataFilterSigmaDiagCov,
                            dataFilterOmegaDiagCov,
                            ))
    print(np.mean(np.array(UKF_result["chi_square"])))
    # np.testing.assert_array_less(np.abs(dataTrueAtt[1:4, :] - dataUKFAtt[1:4]),
    #                              6*np.sqrt(dataUKFAtt[7:10, :]))
    # np.testing.assert_array_less(np.abs(dataTrueAtt[4:7, :] - dataUKFAtt[4:7]),
    #                              6*np.sqrt(dataUKFAtt[10:, :]))
    plot_filter_result_sigma(dataUKFAtt[0,:], dataTrueAtt[1:4, :], dataUKFAtt[1:4, :], dataUKFAtt[7:10, :], dataMeasAtt[1:4, :])
    plot_filter_result_omega(dataUKFAtt[0,:], dataTrueAtt[4:7, :], dataUKFAtt[4:7, :], dataUKFAtt[10:, :])
    plt.show()


if __name__ == "__main__":
    # test_two_sc(
    #     show_plots=True,  # show_plots
    # )
    test_sigma_point_forward_propagation()
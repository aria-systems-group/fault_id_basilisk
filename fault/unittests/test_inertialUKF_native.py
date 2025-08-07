"""
Test if one can use many native forward Basilisk (BSK) simulation to construct UKF - native inertialUKF.
NOTE: for each UKF corresponding to a fault hypothesis, 2*nx+1 number of BSK satellites need to be simulated, in order to simulate the sigma points
Below, we test for two hypothesis {0: nominal, 1: faulty RW}
The fault is injected by adding a viscous friction to the RW1
    - see src/simulation/dynamics/reactionwheels/reactionWheelStateEffector for details of this parameter
The goal is to check if the average chi-square statistics is smaller when the UKF is using the same dynamic as the true.
NOTE: rebuild basilisk since the spacecraft source code is modified.
In brief, the native UKF has the ability to distinguish between nominal and faulty (increases friction) RW.
NOTE: Ideally, we want this test to pass for every dt_obtain_y <= dt_obtain_y', if dt_obtain_y' already passed
However, this is not achievable in the current impelmentation (e.g., if dt_obtain_y = 1.5, it failed)
This is due to the fact that the tuning parameters of UKF: Q actually depend on dt_obtain_y, 
because Q, by definition, is the noise accumulated over each succesive UKF updates.
NOTE: future work should improve the robustness by:
    - advanced unscented transform of sigma points (modifying generate_sigma_points function)
    - advanced UKF such as "square root UKF" to ensure stability (modifying get_ukf_estimates function)
    - Automatic tuning of UKF paraeters (Q, alpha, beta, and kappa)
"""

import os
import pytest
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.linalg import cholesky
from scipy.stats import chi2
warnings.filterwarnings("ignore", category=DeprecationWarning)
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


def plot_filter_result(txt, timeData, state, state_est, cov_est, y_meas=None):
    timeData = timeData/(1E+9)
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for idx, ax in enumerate(axs):
        color = unitTestSupport.getLineColor(idx, 3)
        
        # True state
        ax.plot(timeData[1:], state[1:, idx],
                color=color,
                label=str(txt)+str(idx))
        
        # Estimated state
        ax.plot(timeData[1:], state_est[1:, idx],
                color=color,
                linestyle='--',
                label=str(txt)+str(idx))
        
        if(y_meas is not None):
            # Measured state
            ax.scatter(timeData[1:], y_meas[1:, idx],
                    color="black",
                    label="Measurement "+str(idx)
            )
        
        # ±6 std‐dev band
        std  = 6 * np.sqrt(cov_est[:, idx])
        upper = state_est[:, idx] + std
        lower = state_est[:, idx] - std
        ax.fill_between(timeData[1:], lower[1:], upper[1:],
                        color=color,
                        alpha=0.3,
                        label=r'$\pm6$ std')
        
        # labels & legend
        ax.set_ylabel(str(txt)+str(idx))      # y-label per subplot
        ax.legend(loc='upper right', fontsize='small')

    # Common x‑label on the bottom subplot
    axs[-1].set_xlabel('Time [sec]')


def compare_filter_chisquare(dataChiSquare1, dataChiSquare2):
    p = 0.05      # confidence level
    dof = 3       # degrees of freedom
    chi_ub = chi2.ppf(1-0.5*p, dof)
    chi_lb = chi2.ppf(0.5*p, dof)
    plt.figure()
    plt.scatter(np.arange(len(dataChiSquare1)), dataChiSquare1, color="black", s=10, label=r"data1 - $\chi^2$")
    plt.scatter(np.arange(len(dataChiSquare2)), dataChiSquare2, color="red", s=10, label=r"data2 - $\chi^2$")
    plt.axhline(y=chi_ub, color='cyan', linestyle='--', label=r"$\chi^2$ upper threshold")
    plt.axhline(y=chi_lb, color='b', linestyle='--', label=r"$\chi^2$ lower threshold")
    plt.legend(loc='upper right')


def plot_attitude_error(timeData, dataSigmaBR):
    """Plot the attitude errors."""
    plt.figure()
    for idx in range(3):
        plt.plot(timeData, dataSigmaBR[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\sigma_' + str(idx) + '$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel(r'Attitude Error $\sigma_{B/R}$')


def dict_lists_to_arrays(d: dict) -> dict:
    """
    Convert all list or tuple values in a dictionary into numpy arrays.

    Parameters
    ----------
    d : dict
        Input dictionary whose values are potentially lists/tuples.

    Returns
    -------
    dict
        New dictionary where any list/tuple values have been replaced
        by np.ndarray of the same contents.
    """
    converted = {}
    for key, val in d.items():
        if isinstance(val, (list, tuple)):
            # numpy.array() will handle nested lists as multi-dimensional arrays
            converted[key] = np.array(val)
        else:
            converted[key] = val
    return converted


def generate_sigma_points(x, P, alpha=1e-3, beta=2.0, kappa=1.):
    """
    Generate 2*nx+1 sigma points (Unscented Transform) for state x (nx×1) and covariance P (nx×nx).
    nx = 6: the number of attitude state
    Argurments:
        alpha: [1e-4, 1], tuning parameter of UKF
        kappa: [0, 3*nx], tuning parameter of UKF
    Returns:
        sigma_pts: array[(2*nx+1)×n] of sigma points
        Wm, Wc:   weight vectors for mean and covariance (length 2n+1)
    """
    x = x.reshape(-1,)
    nx   = x.size
    lam = alpha**2*(nx + kappa) - nx
    L = cholesky(P, lower=True)
    sigma_points = [x]
    # weights
    Wm = [lam/(nx+lam)]
    Wc = [lam/(nx+lam) + (1-alpha**2+beta)]
    for i in range(nx):
        sigma_points.append(x + np.sqrt(nx+lam)*L[:, i])
        Wm.append(1./(2.*(nx+lam)))
        Wc.append(1./(2.*(nx+lam)))
    for i in range(nx):
        sigma_points.append(x - np.sqrt(nx+lam)*L[:, i])
        Wm.append(1./(2.*(nx+lam)))
        Wc.append(1./(2.*(nx+lam)))
    sigma_points = np.array(sigma_points)
    Wm = np.array(Wm)
    Wc = np.array(Wc)
    return sigma_points, Wm, Wc


def get_ukf_estimates(y_true, sigma_pts, scLogList, Wm, Wc, n_sigma, Q, R):
    """
    NOTE one should implement MRP subtract instead of linear subtraction used currently.
    """
    # 1) collect propagated sigma-states
    prop = np.zeros_like(sigma_pts)
    for i, log in enumerate(scLogList):
        prop[i, 0:3] = log.sigma_BN[-1, :]
        prop[i, 3:6] = log.omega_BN_B[-1, :]
        # norm_MRP = np.linalg.norm(prop[i, 0:3])
        # if(norm_MRP > 1):
        #     print("[warning] prop switching MRP may be required.")

    # 2) UT-predict mean & covariance
    x_pred = np.dot(Wm, prop).reshape(6,1)
    P_pred = np.zeros((6,6))
    for i in range(n_sigma):
        d = (prop[i, :].reshape(6,1) - x_pred).reshape(6,1)
        P_pred += Wc[i] * (d @ d.T)
    P_pred += Q

    # 3) Regenerate sigma points from (x_pred, P_pred)
    sigma_pts_new, Wm, Wc = generate_sigma_points(x_pred, P_pred)
    # for i in range(sigma_pts_new.shape[0]):
    #     norm_MRP = np.linalg.norm(sigma_pts_new[i, :])
    #     if(norm_MRP > 1):
    #         print("[warning] sigma_pts_new switching MRP may be required.")

    # 4) UT-predicted measurement mean & cov
    Y_sigmas = sigma_pts_new[:, 0:3]
    y_pred = np.dot(Wm, Y_sigmas).reshape(3, 1)
    P_xy  = np.zeros((6,3))
    P_yy = np.zeros((3,3))
    for i in range(n_sigma):
        dx = (sigma_pts_new[i, :].reshape(6,1) - x_pred)
        dy = (Y_sigmas[i, :].reshape(3,1) - y_pred)
        P_xy += Wc[i] * (dx @ dy.T)
        P_yy += Wc[i] * (dy @ dy.T)
    S = P_yy + R

    # 5) Kalman gain & update
    K     = P_xy @ np.linalg.inv(S)
    inno = y_true.reshape(3,1) - y_pred
    x_est = (x_pred + K @ inno).reshape(-1,)
    P_est = P_pred - K @ S @ K.T
    # norm_MRP = np.linalg.norm(x_est[0:3])
    # if(norm_MRP > 1):
    #     print("[warning] x_est switching MRP may be required.")
    return x_est, P_est, inno.reshape(-1,), S


def run_inertialUKF_native(true_Hypo=0, filter_Hypo=0, dt_obtain_y=2.0, random_seed=0, R_level=1.0, show_plots=False):
    """
    Demonstrate UT-prediction: spawn clones for each sigma point,
    propagate them one step, then construct predicted mean & covariance.
    """
    np.random.seed(random_seed)

    # Simulation parameters
    dt_sec = 0.1
    total_time  = 150.0 * dt_obtain_y # to have at least 150 chi_square data          
    steps       = int(total_time / dt_sec)

    # --- Build sim / process / task ---
    sim   = SimulationBaseClass.SimBaseClass()
    proc  = sim.CreateNewProcess("dynProcess")
    dynTaskName = "dynTask"
    task  = sim.CreateNewTask(dynTaskName, macros.sec2nano(dt_sec))
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
    sim.AddModelToTask(dynTaskName, sc_true)
    gravF.addBodiesTo(sc_true)
    scTrueLog = sc_true.scStateOutMsg.recorder()
    sim.AddModelToTask(dynTaskName, scTrueLog)

    # create the FSW vehicle configuration message
    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    vehicleConfigOut.ISCPntB_B = Ivals  # use the same inertia in the FSW algorithm as in the simulation
    vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

    # --- Setup Navigation --- 
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"
    # sNavObject.PMatrix = set_navigation()
    # sNavObject.crossAtt = True
    # connect navigation to spacecraft
    sNavObject.scStateInMsg.subscribeTo(sc_true.scStateOutMsg)
    sim.AddModelToTask(dynTaskName, sNavObject)
    sNavAttLog = sNavObject.attOutMsg.recorder()
    sim.AddModelToTask(dynTaskName, sNavAttLog)

    # --- Setup Reaction Wheels ---
    rwFactory = simIncludeRW.rwFactory()
    varRWModel = messaging.BalancedWheels
    if(true_Hypo == 0):
        RW1 = rwFactory.create('Honeywell_HR16', [1,0,0], maxMomentum=100., Omega=100., RWModel=varRWModel, useMaxTorque=True, useRWfriction=True)
    else:
        RW1 = rwFactory.create('Honeywell_HR16', [1,0,0], maxMomentum=100., Omega=100., RWModel=varRWModel, useMaxTorque=True, useRWfriction=True)
        RW1.cViscous = 3e-3
    print("true RW cViscous friction:", RW1.cViscous)
    RW2 = rwFactory.create('Honeywell_HR16', [0,1,0], maxMomentum=100., Omega=200., RWModel=varRWModel, useMaxTorque=True)
    RW3 = rwFactory.create('Honeywell_HR16', [0,0,1], maxMomentum=100., Omega=300., rWB_B=[0.5,0.5,0.5], RWModel=varRWModel, useMaxTorque=True)
    RW1.Omega_max = RW2.Omega_max
    numRW = rwFactory.getNumOfDevices()
    #--- Connect Reaction Wheels to Spacecraft via rwStateEffector ---
    rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwStateEffector.ModelTag = "RW_cluster"
    rwFactory.addToSpacecraft(sc_true.ModelTag, rwStateEffector, sc_true)
    sim.AddModelToTask(dynTaskName, rwStateEffector)
    # setup inertial3D guidance module
    inertial3DObj = inertial3D.inertial3D()
    inertial3DObj.ModelTag = "inertial3D"
    sim.AddModelToTask(dynTaskName, inertial3DObj)
    inertial3DObj.sigma_R0N = [0., 0., 0.]  # set the desired inertial orientation
    # setup the attitude tracking error evaluation module
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    sim.AddModelToTask(dynTaskName, attError)
    attErrorLog = attError.attGuidOutMsg.recorder()
    sim.AddModelToTask(dynTaskName, attErrorLog)
    # setup the MRP Feedback control module
    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedback"
    sim.AddModelToTask(dynTaskName, mrpControl)
    mrpControl.K = 3.5
    mrpControl.Ki = -1  # make value negative to turn off integral feedback
    mrpControl.P = 30.0
    mrpControl.integralLimit = 2. / mrpControl.Ki * 0.1
    # --- Connect messages ---
    # create the FSW reaction wheel configuration message
    fswRwParamMsg = rwFactory.getConfigMessage()
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
    sim.AddModelToTask(dynTaskName, rwMotorTorqueObj)
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

    # --- Setup 3D Visualiztion ---
    if(show_plots):
        viz = vizSupport.enableUnityVisualization(sim, dynTaskName, sc_true
                                                , saveFile=fileName
                                                , rwEffectorList=rwStateEffector
                                                )

    #####################################################
    # --- Setup S/C clones for UKF ---
    
    # setup initial state and covariance
    dx0 = np.array([0.3, 0.5, 0.1, 0., 0., 0.])
    x_est = x0 + dx0
    P_est = np.eye(6) * 1.0
    UKF_result = {
        "timeNanos": [0],
        "x_est": [x_est],
        "P_est": [P_est],
        "inno": [np.full(3, np.nan)],
        "S": [np.zeros((3, 3))],
        "chi_square": [],
    }
    Measure = {
        "timeNanos": [0],
        "y": [np.full(3, np.nan)]
    }
    Q_accel = np.diag([1e-8, 1e-8, 1e-8])
    Gamma = np.vstack((np.diag([0.5*dt_obtain_y**2,
                                0.5*dt_obtain_y**2,
                                0.5*dt_obtain_y**2 ]), 
                       np.diag([dt_obtain_y, dt_obtain_y, dt_obtain_y])))
    Q     = Gamma @ Q_accel @ Gamma.T
    R     = np.eye(3)*1e-3 * R_level
    
    # setup S/C clones vehicle
    vc_payload_i = messaging.VehicleConfigMsgPayload()
    vc_payload_i.ISCPntB_B = Ivals
    vc_i = messaging.VehicleConfigMsg().write(vc_payload_i)

    # setup sigma-point
    sigma_pts, Wm, Wc = generate_sigma_points(x_est, P_est)
    n_sigma = sigma_pts.shape[0]

    # setup reaction wheel configuration
    rw_factory_list = []
    rw_config_msg_list = []
    for i in range(n_sigma):
        rw_factory_i = simIncludeRW.rwFactory()
        rw_model_i   = messaging.BalancedWheels
        if(filter_Hypo == 0):
            rw1_i =rw_factory_i.create('Honeywell_HR16', [1,0,0], maxMomentum=100., Omega=100., RWModel=rw_model_i, useMaxTorque=True, useRWfriction=True)
        else:
            rw1_i =rw_factory_i.create('Honeywell_HR16', [1,0,0], maxMomentum=100., Omega=100., RWModel=rw_model_i, useMaxTorque=True, useRWfriction=True)
            rw1_i.cViscous = 3e-3
        if(i == 0):
            print("filter rw cViscous friction: ", rw1_i.cViscous)
        rw2_i = rw_factory_i.create('Honeywell_HR16', [0,1,0], maxMomentum=100., Omega=200., RWModel=rw_model_i, useMaxTorque=True)
        rw_factory_i.create('Honeywell_HR16', [0,0,1], maxMomentum=100., Omega=300., rWB_B=[0.5,0.5,0.5], RWModel=rw_model_i, useMaxTorque=True)
        rw1_i.Omega_max = rw2_i.Omega_max
        rw_factory_list.append(rw_factory_i)
        rw_config_msg_list.append(rw_factory_i.getConfigMessage())

    # create S/C clones
    scList, scLogList = [], []    
    for i in range(n_sigma):
        sc_i = spacecraft.Spacecraft()
        sc_i.ModelTag = f"scSigma_{i}"
        sc_i.hub.mHub           = 750.0
        sc_i.hub.r_BcB_B        = [[0.0],[0.0],[0.0]]
        sc_i.hub.IHubPntBc_B    = unitTestSupport.np2EigenMatrix3d(Ivals)
        sc_i.hub.r_CN_NInit     = np.copy(r0)
        sc_i.hub.v_CN_NInit     = np.copy(v0)
        # set attitude+rate from sigma point
        sc_i.hub.sigma_BNInit   = sigma_pts[i,0:3].reshape(3,1)
        sc_i.hub.omega_BN_BInit = sigma_pts[i,3:6].reshape(3,1)
        
        # add to sim & gravity
        sim.AddModelToTask(dynTaskName, sc_i)
        gravF.addBodiesTo(sc_i)
        log_i = sc_i.scStateOutMsg.recorder()
        sim.AddModelToTask(dynTaskName, log_i)
        scList.append(sc_i)
        scLogList.append(log_i)

        # navigation
        nav_i = simpleNav.SimpleNav()
        nav_i.ModelTag = "Nav"+str(i)
        nav_i.scStateInMsg.subscribeTo(sc_i.scStateOutMsg)
        sim.AddModelToTask(dynTaskName, nav_i)

        # rw effector connect to rw configuration
        rw_eff_i = reactionWheelStateEffector.ReactionWheelStateEffector()
        rw_eff_i.ModelTag = "RWEff"+str(i)
        rw_factory_list[i].addToSpacecraft(sc_i.ModelTag, rw_eff_i, sc_i)
        sim.AddModelToTask(dynTaskName, rw_eff_i)

        # guidance & control
        guid = inertial3D.inertial3D();        guid.ModelTag = "Guid"+str(i); sim.AddModelToTask(dynTaskName, guid); guid.sigma_R0N = [0,0,0]
        err  = attTrackingError.attTrackingError(); err.ModelTag  = "Err"+str(i); sim.AddModelToTask(dynTaskName, err)
        ctrl = mrpFeedback.mrpFeedback();       ctrl.ModelTag = "Ctrl"+str(i); sim.AddModelToTask(dynTaskName, ctrl)
        motor= rwMotorTorque.rwMotorTorque();  motor.ModelTag= "Motor"+str(i); sim.AddModelToTask(dynTaskName, motor)

        # set controller gains
        ctrl.K  = 3.5
        ctrl.Ki = -1.0
        ctrl.P  = 30.0
        ctrl.integralLimit = 2.0/ctrl.Ki*0.1

        # connect messages
        # nav → error
        err.attNavInMsg.subscribeTo(nav_i.attOutMsg)
        # guid → error
        err.attRefInMsg.subscribeTo(guid.attRefOutMsg)
        # error → controller
        ctrl.guidInMsg.subscribeTo(err.attGuidOutMsg)
        # vehicle config → controller
        ctrl.vehConfigInMsg.subscribeTo(vc_i)
        # RW params/speeds → controller
        ctrl.rwParamsInMsg.subscribeTo(rw_config_msg_list[i])
        ctrl.rwSpeedsInMsg.subscribeTo(rw_eff_i.rwSpeedOutMsg)
        # controller → motor
        motor.controlAxes_B   = [1,0,0, 0,1,0, 0,0,1]
        motor.rwParamsInMsg   .subscribeTo(rw_config_msg_list[i])
        motor.vehControlInMsg .subscribeTo(ctrl.cmdTorqueOutMsg)
        # RW effector → motor torque
        rw_eff_i.rwMotorCmdInMsg.subscribeTo(motor.rwMotorTorqueOutMsg)


    #########################################
    # --- Simulation Time Loop ---

    sim.InitializeSimulation()
    current_time = 0.0
    ytrue = None
    for kk in range(steps):
        # propagate true & clones one step
        current_time += dt_sec
        sim.ConfigureStopTime(macros.sec2nano(current_time))
        sim.ExecuteSimulation()

        # obtain attitude sigma_BN measurements
        if(macros.sec2nano(current_time) % macros.sec2nano(dt_obtain_y) == 0):
            Measure["timeNanos"].append(macros.sec2nano(current_time))
            ytrue = scTrueLog.sigma_BN[-1, :] + np.random.normal(0, np.sqrt(R[0,0]), 3)
            Measure["y"].append(ytrue)
        else:
            ytrue = None

        # do UKF by S/C clone simulation
        if(ytrue is not None):
            # get UKF state and covariance estimates
            x_est, P_est, inno, S = get_ukf_estimates(ytrue, sigma_pts, scLogList, Wm, Wc, n_sigma, Q, R)
            # record UKF results
            UKF_result["timeNanos"].append(macros.sec2nano(current_time))
            UKF_result["x_est"].append(x_est)
            UKF_result["P_est"].append(P_est)
            UKF_result["inno"].append(inno)
            UKF_result["S"].append(S)
            
            # Update the attitude state in each S/C clone from UKF estimates
            sigma_pts, Wm, Wc = generate_sigma_points(x_est, P_est)
            for idx, sc_i in enumerate(scList):
                sigma_est = np.copy(sigma_pts[idx, 0:3]) #scTrueLog.sigma_BN[-1]
                omega_BN_B_est = np.copy(sigma_pts[idx, 3:6]) #scTrueLog.omega_BN_B[-1]
                sc_i.setSigmaBN(sigma_est)
                sc_i.setOmegaBN_B(omega_BN_B_est)

    #########################################
    # --- Post processing & Checking ---
    timeData = scTrueLog.times()
    dataSigmaBR = attErrorLog.sigma_BR
    plot_attitude_error(timeData, dataSigmaBR)

    # prepare data
    UKF_result = dict_lists_to_arrays(UKF_result)
    Measure = dict_lists_to_arrays(Measure)
    dataFilterSigmaDiagCov = np.diagonal(UKF_result["P_est"], 
                                         axis1=1, axis2=2)[:, :3]
    dataFilterOmegaDiagCov = np.diagonal(UKF_result["P_est"], 
                                         axis1=1, axis2=2)[:, 3:6]
    timeDataUKF = UKF_result["timeNanos"]
    mask     = np.isin(timeData, timeDataUKF)        # [False  True False  True  True]
    idx      = np.nonzero(mask)[0]  # array([1, 3, 4])
    dataTrueSigmaBN = scTrueLog.sigma_BN[idx, :]
    dataTrueOmegaBN = scTrueLog.omega_BN_B[idx, :]
    # print("[debug]", timeDataUKF.shape, dataTrueSigmaBN.shape, UKF_result["x_est"][:, 0:3].shape, dataFilterSigmaDiagCov.shape, Measure["y"].shape)

    # if the filter is consistent with the true, check the UKF state and covariane estimates
    plot_filter_result("sigma", timeDataUKF, dataTrueSigmaBN, UKF_result["x_est"][:, 0:3], dataFilterSigmaDiagCov, Measure["y"])
    plot_filter_result("omega", timeDataUKF, dataTrueOmegaBN, UKF_result["x_est"][:, 3:6], dataFilterOmegaDiagCov)
    if(true_Hypo == filter_Hypo):
        np.testing.assert_array_less(np.abs(dataTrueSigmaBN - UKF_result["x_est"][:, 0:3])[-10:, :], 
                                 6*np.sqrt(dataFilterSigmaDiagCov)[-10:, :])
        np.testing.assert_array_less(np.abs(dataTrueOmegaBN - UKF_result["x_est"][:, 3:6])[-10:, :], 
                                 6*np.sqrt(dataFilterOmegaDiagCov)[-10:, :])

    # Collect statistic data needed for Hypothesis Identification ---
    dataChiSquare = []
    for i in range(UKF_result["inno"].shape[0]):
        inno_i = UKF_result["inno"][i, :].reshape(3, 1)
        S_i = UKF_result["S"][i, :, :]
        if(any(np.isnan(inno_i))):
            dataChiSquare.append(np.nan)
        else:
            chi_sqaure = (inno_i.T @ np.linalg.inv(S_i) @ inno_i)
            dataChiSquare.append(chi_sqaure.item())
    dataChiSquare = np.array(dataChiSquare)
    valid = dataChiSquare[~np.isnan(dataChiSquare)]
    mean = valid.mean()
    print("chi-square avg [under true hypo {:2d} and filter hypo {:2d}]: {:.5f}".format(true_Hypo, filter_Hypo, mean))
    print("=====")

    if(show_plots):
        plt.show()
    plt.close("all")

    return mean, dataChiSquare


@pytest.mark.parametrize("R_level", [1.0, 0.1])
@pytest.mark.parametrize("random_seed", [0, 1, 2])
@pytest.mark.parametrize("dt_obtain_y", [2.0, 2.5])
def test_inertialUKF_native_statistics(dt_obtain_y, random_seed, R_level):
    chi_square_correct_UKF, _ = run_inertialUKF_native(
        true_Hypo=0,
        filter_Hypo=0,
        dt_obtain_y=dt_obtain_y,
        random_seed=random_seed,
        R_level=R_level,
        show_plots=False,  
    )
    chi_square_incorrect_UKF, _ = run_inertialUKF_native(
        true_Hypo=0,
        filter_Hypo=1,
        dt_obtain_y=dt_obtain_y,
        random_seed=random_seed,
        R_level=R_level,
        show_plots=False,  
    )
    np.testing.assert_array_less(chi_square_correct_UKF, chi_square_incorrect_UKF)

    chi_square_correct_UKF, _ = run_inertialUKF_native(
        true_Hypo=1,
        filter_Hypo=1,
        dt_obtain_y=dt_obtain_y,
        random_seed=random_seed,
        R_level=R_level,
        show_plots=False,  
    )
    chi_square_incorrect_UKF, _ = run_inertialUKF_native(
        true_Hypo=1,
        filter_Hypo=0,
        dt_obtain_y=dt_obtain_y,
        random_seed=random_seed,
        R_level=R_level,
        show_plots=False,  
    )
    np.testing.assert_array_less(chi_square_correct_UKF, chi_square_incorrect_UKF)


if __name__ == "__main__":
    _, data_chi_square_correct = run_inertialUKF_native(true_Hypo=0, filter_Hypo=0, show_plots=True)
    _, data_chi_square_inccorrect = run_inertialUKF_native(true_Hypo=0, filter_Hypo=1, show_plots=True)
    compare_filter_chisquare(data_chi_square_correct, data_chi_square_inccorrect)
    plt.show()
    plt.close("all")

    test_inertialUKF_native_statistics(dt_obtain_y=2.0, random_seed=0, R_level=1.0)
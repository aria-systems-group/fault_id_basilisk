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
from numpy.linalg import cholesky

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


def plot_filter_result(txt, timeData, state, state_est, cov_est, y_meas=None):
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for idx, ax in enumerate(axs):
        color = unitTestSupport.getLineColor(idx, 3)
        
        # True state
        ax.plot(timeData, state[:, idx],
                color=color,
                label=str(txt)+str(idx))
        
        # Estimated state
        ax.plot(timeData, state_est[:, idx],
                color=color,
                linestyle='--',
                label=str(txt)+str(idx))
        
        if(y_meas is not None):
            # Measured state
            ax.scatter(timeData, y_meas[:, idx],
                    color="black",
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
    axs[-1].set_xlabel('Time [min]')


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


def generate_sigma_points(x, P):
    """
    Generate 2n+1 sigma points (Unscented Transform) for state x (n×1) and covariance P (n×n).
    Returns:
        sigma_pts: array[(2n+1)×n] of sigma points
        Wm, Wc:   weight vectors for mean and covariance (length 2n+1)
    """
    nx   = x.size
    lam = 2.0
    L = np.linalg.cholesky((nx + lam) * P)
    sigma_points = [x]
    # weights
    Wm = [lam/(nx+lam)]
    for i in range(nx):
        sigma_points.append(x + L[:, i])
        Wm.append(1./(2.*(nx+lam)))
    for i in range(nx):
        sigma_points.append(x - L[:, i])
        Wm.append(1./(2.*(nx+lam)))
    Wc = Wm
    sigma_points = np.array(sigma_points)
    Wm = np.array(Wm)
    Wc = np.array(Wc)
    return sigma_points, Wm, Wc


def get_ukf_estimates(y_true, sigma_pts, scLogList, Wm, Wc, n_sigma, Q, R):
    # 1) collect propagated sigma-states
    prop = np.zeros_like(sigma_pts)
    for i, log in enumerate(scLogList):
        prop[i, 0:3] = log.sigma_BN[-1, :]
        prop[i, 3:6] = log.omega_BN_B[-1, :]
        norm_MRP = np.linalg.norm(prop[i, 0:3])
        if(norm_MRP > 1):
            print(norm_MRP)

    # 2) UT-predict mean & covariance
    x_pred = np.dot(Wm, prop).reshape(6,1)
    P_pred = np.zeros((6,6))
    for i in range(n_sigma):
        d = (prop[i, :].reshape(6,1) - x_pred).reshape(6,1)
        P_pred += Wc[i] * (d @ d.T)
    P_pred += Q

    # 4) UT-predicted measurement mean & cov
    Y_sigmas = prop[:, 0:3]
    y_pred = np.dot(Wm, Y_sigmas).reshape(3, 1)
    P_xy  = np.zeros((6,3))
    P_yy = np.zeros((3,3))
    for i in range(n_sigma):
        dx = (prop[i, :].reshape(6,1) - x_pred)
        dy = (Y_sigmas[i, :].reshape(3,1) - y_pred)
        P_xy += Wc[i] * (dx @ dy.T)
        P_yy += Wc[i] * (dy @ dy.T)
    S = P_yy + R

    # 5) Kalman gain & update
    K     = P_xy @ np.linalg.inv(S)
    inno = y_true.reshape(3,1) - y_pred
    x_est = (x_pred + K @ (inno)).reshape(-1,)
    P_est = P_pred - K @ S @ K.T
    return x_est, P_est, inno.reshape(-1,), S


def run_inertialUKF_native(true_Hypo=0, filter_Hypo=0, show_plots=False):
    """
    Demonstrate UT-prediction: spawn clones for each sigma point,
    propagate them one step, then reconstruct predicted mean & covariance.
    """
    np.random.seed(42)

    # Simulation parameters
    dt_sec      = 1.
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
    P0 = np.eye(6) * 1.0
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
    sim.AddModelToTask("task", sc_true)
    gravF.addBodiesTo(sc_true)
    scTrueLog = sc_true.scStateOutMsg.recorder()
    sim.AddModelToTask("task", scTrueLog)

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
    sim.AddModelToTask("task", sNavObject)
    sNavAttLog = sNavObject.attOutMsg.recorder()
    sim.AddModelToTask("task", sNavAttLog)

    # --- Setup Reaction Wheels ---
    rwFactory = simIncludeRW.rwFactory()
    varRWModel = messaging.BalancedWheels
    if(true_Hypo == 0):
        RW1 = rwFactory.create('Honeywell_HR16', [1,0,0], maxMomentum=100., Omega=100., RWModel=varRWModel, useMaxTorque=True)
    else:
        RW1 = rwFactory.create('Honeywell_HR16', [1,0,0], maxMomentum=100., Omega=100., Omega_max=500., RWModel=varRWModel, useMaxTorque=True)
    RW2 = rwFactory.create('Honeywell_HR16', [0,1,0], maxMomentum=100., Omega=200., RWModel=varRWModel, useMaxTorque=True)
    RW3 = rwFactory.create('Honeywell_HR16', [0,0,1], maxMomentum=100., Omega=300., rWB_B=[0.5,0.5,0.5], RWModel=varRWModel, useMaxTorque=True)
    RW1.Omega_max = RW2.Omega_max
    # RW1.Omega_max = RW2.Omega_max
    numRW = rwFactory.getNumOfDevices()
    #--- Connect Reaction Wheels to Spacecraft via rwStateEffector ---
    rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwStateEffector.ModelTag = "RW_cluster"
    rwFactory.addToSpacecraft(sc_true.ModelTag, rwStateEffector, sc_true)
    # add RW object array to the simulation process.  This is required for the UpdateState() method
    # to be called which logs the RW states
    sim.AddModelToTask("task", rwStateEffector)
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

    #####################################################

    # --- UKF init ---
    dx0 = np.array([0.3, 0.5, 0.1, 0., 0., 0.])
    x_est = x0 + dx0
    P_est = P0
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
    Q     = np.diag([1e-8, 1e-8, 1e-8, 1e-7, 1e-7, 1e-7])
    R     = np.eye(3)*1e-3  # measurement noise on σ only

    # — Vehicle config message —
    vc_payload_i = messaging.VehicleConfigMsgPayload()
    vc_payload_i.ISCPntB_B = Ivals
    vc_i = messaging.VehicleConfigMsg().write(vc_payload_i)

    # spawn initial sigma-point clones
    sigma_pts, Wm, Wc = generate_sigma_points(x_est, P_est)
    n_sigma = sigma_pts.shape[0]

    # --- Prepare S/C UKF controller ---
    rw_factory_list = []
    rw_config_msg_list = []
    for i in range(n_sigma):
        # — Reaction wheels factory & devices —
        rw_factory_i = simIncludeRW.rwFactory()
        rw_model_i   = messaging.BalancedWheels
        if(filter_Hypo == 0):
            rw1_i =rw_factory_i.create('Honeywell_HR16', [1,0,0], maxMomentum=100., Omega=100., RWModel=rw_model_i, useMaxTorque=True)
        else:
            rw1_i =rw_factory_i.create('Honeywell_HR16', [1,0,0], maxMomentum=100., Omega=100., RWModel=rw_model_i, Omega_max=500., useMaxTorque=True)
        rw2_i = rw_factory_i.create('Honeywell_HR16', [0,1,0], maxMomentum=100., Omega=200., RWModel=rw_model_i, useMaxTorque=True)
        rw_factory_i.create('Honeywell_HR16', [0,0,1], maxMomentum=100., Omega=300., rWB_B=[0.5,0.5,0.5], RWModel=rw_model_i, useMaxTorque=True)
        rw1_i.Omega_max = rw2_i.Omega_max
        rw_factory_list.append(rw_factory_i)
        rw_config_msg_list.append(rw_factory_i.getConfigMessage())

    scList, scLogList = [], []    
    for i in range(n_sigma):
        sc_i = spacecraft.Spacecraft()
        sc_i.ModelTag = f"scSigma_{i}"
        sc_i.hub.mHub           = 750.0
        sc_i.hub.r_BcB_B        = [[0.0],[0.0],[0.0]]
        sc_i.hub.IHubPntBc_B    = unitTestSupport.np2EigenMatrix3d(Ivals)
        sc_i.hub.r_CN_NInit     = np.copy(r0)
        sc_i.hub.v_CN_NInit     = np.copy(v0)
        # set attitude+rate from sigma point idx
        sc_i.hub.sigma_BNInit   = sigma_pts[i,0:3].reshape(3,1)
        sc_i.hub.omega_BN_BInit = sigma_pts[i,3:6].reshape(3,1)
        
        # add to sim & gravity
        sim.AddModelToTask("task", sc_i)
        gravF.addBodiesTo(sc_i)
        log_i = sc_i.scStateOutMsg.recorder()
        sim.AddModelToTask("task", log_i)
        scList.append(sc_i)
        scLogList.append(log_i)

        # — Navigation —
        nav_i = simpleNav.SimpleNav()
        nav_i.ModelTag = "Nav"+str(i)
        nav_i.scStateInMsg.subscribeTo(sc_i.scStateOutMsg)
        sim.AddModelToTask("task", nav_i)

        # — RW effector hookup —
        rw_eff_i = reactionWheelStateEffector.ReactionWheelStateEffector()
        rw_eff_i.ModelTag = "RWEff"+str(i)
        rw_factory_list[i].addToSpacecraft(sc_i.ModelTag, rw_eff_i, sc_i)
        sim.AddModelToTask("task", rw_eff_i)

        # — Guiding & control chain —
        guid = inertial3D.inertial3D();        guid.ModelTag = "Guid"+str(i); sim.AddModelToTask("task", guid); guid.sigma_R0N = [0,0,0]
        err  = attTrackingError.attTrackingError(); err.ModelTag  = "Err"+str(i); sim.AddModelToTask("task", err)
        ctrl = mrpFeedback.mrpFeedback();       ctrl.ModelTag = "Ctrl"+str(i); sim.AddModelToTask("task", ctrl)
        motor= rwMotorTorque.rwMotorTorque();  motor.ModelTag= "Motor"+str(i); sim.AddModelToTask("task", motor)

        # — Set controller gains —
        ctrl.K  = 3.5
        ctrl.Ki = -1.0
        ctrl.P  = 30.0
        ctrl.integralLimit = 2.0/ctrl.Ki*0.1

        # — Message wiring —
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

        # # controller → motor
        motor.controlAxes_B   = [1,0,0, 0,1,0, 0,0,1]
        motor.rwParamsInMsg   .subscribeTo(rw_config_msg_list[i])
        motor.vehControlInMsg .subscribeTo(ctrl.cmdTorqueOutMsg)

        # RW effector → motor torque
        rw_eff_i.rwMotorCmdInMsg.subscribeTo(motor.rwMotorTorqueOutMsg)


    #########################################

    # --- Time loop ---
    sim.InitializeSimulation()
    current_time = 0.0
    ytrue = None
    for kk in range(steps):
        # propagate true & clones one step
        current_time += dt_sec
        sim.ConfigureStopTime(macros.sec2nano(current_time))
        sim.ExecuteSimulation()

        Measure["timeNanos"].append(macros.sec2nano(current_time))
        ytrue = scTrueLog.sigma_BN[-1, :] + np.random.normal(0, np.sqrt(1e-3), 3)

        if(ytrue is not None):
            Measure["y"].append(ytrue)
            # print("x_est:", " ".join("{:.1f}".format(v) for v in x_est))
            # Do UKF (See test_sigma_point_forward_propagation)
            x_est, P_est, inno, S = get_ukf_estimates(ytrue, sigma_pts, scLogList, Wm, Wc, n_sigma, Q, R)
            UKF_result["timeNanos"].append(macros.sec2nano(current_time))
            UKF_result["x_est"].append(x_est)
            UKF_result["P_est"].append(P_est)
            UKF_result["inno"].append(inno)
            UKF_result["S"].append(S)
            
            # Update the state in each sc_i from UKF estimates
            sigma_pts, Wm, Wc = generate_sigma_points(x_est, P_est)
            for idx, sc_i in enumerate(scList):
                sigma_est = np.copy(sigma_pts[idx, 0:3]) #scTrueLog.sigma_BN[-1]
                omega_BN_B_est = np.copy(sigma_pts[idx, 3:6]) #scTrueLog.omega_BN_B[-1]
                sc_i.setSigmaBN(sigma_est)
                sc_i.setOmegaBN_B(omega_BN_B_est)
        else:
            Measure["y"].append(np.nan)
            

    #########################################

    UKF_result = dict_lists_to_arrays(UKF_result)
    Measure = dict_lists_to_arrays(Measure)
    # print(UKF_result["x_est"].shape, scTrueLog.sigma_BN.shape, Measure["y"].shape, UKF_result["P_est"].shape,
    #       UKF_result["inno"].shape, UKF_result["S"].shape)

    dataFilterSigmaDiagCov = np.diagonal(UKF_result["P_est"], 
                                         axis1=1, axis2=2)[:, :3]
    dataFilterOmegaDiagCov = np.diagonal(UKF_result["P_est"], 
                                         axis1=1, axis2=2)[:, 3:6]
    timeData = scTrueLog.times() * macros.NANO2MIN

    check_time = int(total_time*0.5)
    if(true_Hypo == filter_Hypo):
        np.testing.assert_array_less(np.abs(scTrueLog.sigma_BN - UKF_result["x_est"][:, 0:3])[check_time:, :], 
                                 6*np.sqrt(dataFilterSigmaDiagCov)[check_time:, :])
        np.testing.assert_array_less(np.abs(scTrueLog.omega_BN_B - UKF_result["x_est"][:, 3:6])[check_time:, :], 
                                 6*np.sqrt(dataFilterOmegaDiagCov)[check_time:, :])

    plot_filter_result("sigma", timeData, scTrueLog.sigma_BN, UKF_result["x_est"][:, 0:3], dataFilterSigmaDiagCov, Measure["y"])
    plot_filter_result("omega", timeData, scTrueLog.omega_BN_B, UKF_result["x_est"][:, 3:6], dataFilterOmegaDiagCov)

    # --- Statistic data needed for Hypothesis Identification ---
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
    print("chi-square avg [under true hypo {:2d} and filter hypo {:2d}]: {:.4f}".format(true_Hypo, filter_Hypo, mean))

    if(show_plots):
        plt.show()

    return mean


def test_inertialUKF_native_statistics():
    chi_square_correct_UKF = run_inertialUKF_native(
        true_Hypo=0,
        filter_Hypo=0,
        show_plots=False,  
    )
    chi_square_incorrect_UKF = run_inertialUKF_native(
        true_Hypo=0,
        filter_Hypo=1,
        show_plots=False,  
    )
    np.testing.assert_array_less(chi_square_correct_UKF, chi_square_incorrect_UKF)

    chi_square_correct_UKF = run_inertialUKF_native(
        true_Hypo=1,
        filter_Hypo=1,
        show_plots=False,  
    )
    chi_square_incorrect_UKF= run_inertialUKF_native(
        true_Hypo=1,
        filter_Hypo=0,
        show_plots=False,  
    )
    np.testing.assert_array_less(chi_square_correct_UKF, chi_square_incorrect_UKF)


if __name__ == "__main__":
    _ = run_inertialUKF_native(show_plots=True)

    test_inertialUKF_native_statistics()
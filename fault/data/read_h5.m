% Read data
clc; clear; close;

filename = 'healthy_rw_run.h5';
% filename = 'faulty_rw_run.h5';

% Display the contents of the file
h5disp(filename);

% Read top-level datasets
labels = h5read(filename, '/labels');
time_s = h5read(filename, '/time_s');

% Read sensor datasets
attitude_mrp       = h5read(filename, '/sensors/attitude_mrp');
body_rates_radps   = h5read(filename, '/sensors/body_rates_radps');
rw_motor_torque_Nm = h5read(filename, '/sensors/rw_motor_torque_Nm');
rw_wheel_speed_radps = h5read(filename, '/sensors/rw_wheel_speed_radps'); 

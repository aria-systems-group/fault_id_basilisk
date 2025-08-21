% Read data
clc; clear; close;

filename1 = 'healthy_rw_run.h5';
filename2 = 'faulty_rw_run.h5';

% Display the contents of the file
h5disp(filename1);
h5disp(filename2);

% Read top-level datasets
labels1 = h5read(filename1, '/labels');
time_s1 = h5read(filename1, '/time_s');

labels2 = h5read(filename2, '/labels');
time_s2 = h5read(filename2, '/time_s');

% Read sensor datasets
attitude_mrp1       = h5read(filename1, '/sensors/attitude_mrp');
body_rates_radps1   = h5read(filename1, '/sensors/body_rates_radps');
rw_motor_torque_Nm1 = h5read(filename1, '/sensors/rw_motor_torque_Nm');
rw_wheel_speed_radps1 = h5read(filename1, '/sensors/rw_wheel_speed_radps'); 

attitude_mrp2       = h5read(filename2, '/sensors/attitude_mrp');
body_rates_radps2   = h5read(filename2, '/sensors/body_rates_radps');
rw_motor_torque_Nm2 = h5read(filename2, '/sensors/rw_motor_torque_Nm');
rw_wheel_speed_radps2 = h5read(filename2, '/sensors/rw_wheel_speed_radps'); 

% Difference
diff_att = max(max(attitude_mrp2 - attitude_mrp1))
diff_body = max(max(body_rates_radps2 - body_rates_radps1))
diff_rw_motor = max(max(rw_motor_torque_Nm2 - rw_motor_torque_Nm1))
diff_rw_wheel = max(max(rw_wheel_speed_radps2 - rw_wheel_speed_radps1))
diff_labels = max(max(labels2 - labels1))


% Figure
figure(1)
plot(time_s1, attitude_mrp1)

figure(2)
plot(time_s1, attitude_mrp2)
















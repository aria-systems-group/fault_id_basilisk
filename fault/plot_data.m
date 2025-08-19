clear; clc;

% Load the data
data = load('spacecraft_nominal_vs_faulty.mat');

t = data.time;
fault_time = data.fault_time;

% Euler angles
euler_nom = data.euler_nominal;  % [yaw pitch roll] for nominal case
euler_fault = data.euler_faulty;

% Angular velocities
omega_nom = data.omega_nominal;  % rad/s
omega_fault = data.omega_faulty;

% Convert radians to degrees for better readability
euler_nom_deg = rad2deg(euler_nom);
euler_fault_deg = rad2deg(euler_fault);
omega_nom_deg = rad2deg(omega_nom);
omega_fault_deg = rad2deg(omega_fault);

%% Plot Euler angles
figure;
titles = {'Yaw (ψ)', 'Pitch (θ)', 'Roll (φ)'};
for i = 1:3
    subplot(3,1,i);
    plot(t, euler_nom_deg(:,i), 'b', 'LineWidth', 1.5); hold on;
    plot(t, euler_fault_deg(:,i), 'r--', 'LineWidth', 1.5);
    xline(fault_time, 'k--', 'Fault Time');
    ylabel('deg');
    title(['Euler Angle: ', titles{i}]);
    grid on;
end
xlabel('Time [s]');
legend('Nominal', 'Faulty');

%% Plot Angular Velocities
figure;
titles = {'\omega_x', '\omega_y', '\omega_z'};
for i = 1:3
    subplot(3,1,i);
    plot(t, omega_nom_deg(:,i), 'b', 'LineWidth', 1.5); hold on;
    plot(t, omega_fault_deg(:,i), 'r--', 'LineWidth', 1.5);
    xline(fault_time, 'k--', 'Fault Time');
    ylabel('deg/s');
    title(['Angular Velocity: ', titles{i}]);
    grid on;
end
xlabel('Time [s]');
legend('Nominal', 'Faulty');

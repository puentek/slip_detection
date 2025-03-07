close all

clc

data = readtable('finger_push_test1.csv');

[rows, columns] = size(data); 

num_sensors = 8; % number of sensors

sensor_data = ones(rows, num_sensors); % pre-allocated sensor data array
time = 0.1 .* linspace(0, rows, 475);

figure()

for i = 1:8

    sensor_data(:, i) = table2array(data(:, i));
    
    % plot(sensor_data{i})
    % hold on
    plot(time', sensor_data(:, i))
    hold on

end

xlabel('Time [s]');
ylabel('Pressure [mBar]');

p_psi_max = max(sensor_data) .* 0.0145038; % convert pressure units from mBar to psi

p_psi_min = min(sensor_data) .* 0.0145038;
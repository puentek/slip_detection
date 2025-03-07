clear all
close all


clc

data = readtable('C:\Users\18328\Desktop\onr_tact\ONR_data\weight_test_all_sensor_raw_calibrated.csv', "VariableNamingRule", "preserve");

% legend_labels = {'Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8'};

for i = 1:8
    

    pressure_sensor = table2array(data(:, i)); % Pressure Reading

    num_samples = length(pressure_sensor);
    time = (0:num_samples-1) / 10; % Time in seconds

    p_in_psi = pressure_sensor .* 0.0145038;

    % plot(time, pressure_sensor)
    % % figure()
    plot(time, p_in_psi, 'LineWidth', 1.4)

    hold on

    % hold on
    % % title(sprintf('Sensor %d', i))

    % file_name = sprintf("Sensor_all.png");
    % save_path = fullfile('.\Plot Images', file_name);

    % exportgraphics(gcf, save_path, 'Resolution', 250);
    

end

title('Calibrated Data')
legend('Location', 'eastoutside')
clear; clc; clear all; 

% define the folders we will use for this code aka train and test folder
train_folder = "C:/Users/18328/Desktop/onr_tact/converted_data_for_slip/train_folder"
test_folder = "C:\Users\18328\Desktop\onr_tact\converted_data_for_slip\test_folder"

% lets load the csv files we will be using 
train_files = dir(fullfile(train_folder, '*.csv'));
test_files = dir(fullfile(test_folder, '*.csv'));

%  check if the number of files we said we have matches 
if lenth(train_files) < 8 || length(test_files) < 2
    error("Expected 8 training files and 2 test files. Check again ")
end 

% Define the window and step size (this can be adjusted)

% number of samples per window 
window_size = 10;
% step size to slide the window
step_size = 5;

% Initialize training data 
X_train =  [];
y_train = [];

%  load and process the data 
for i = 1:8
    % load file 
    data = readtable(fullfile(train_folder,train_files(i).name));

    % extracting the time and sensor values 
    time = data.("Time_s_");
    % Extract sensors 1 -8
    sensor_values = table2array(data(:,2:9)); 

    % define slip condition (this will change after ground truth)
    % slip occurs if sensor 8 <0.6 (low pressure on Sensor 8)
    % or if any sensors 1-7 are > 0.5 (high pressure on other sensors)
    slip_labels = (data.Sensor8 <0.6) | any(sensor_values(:,1:7) > 0.5, 2);

    %  now we apply the moving window 
    num_samples = length(time):
        for j = 1:step_size:(num_samples-window_size+1)
            % extract window data 
            window_data = sensor_values(j:j+window_size-1,:);
            % majority class in window 
            window_label = mode(slip_labels(j:j+window_size-1));

            % compute window based features (mean, std, max, min)
            windo_features = [mean(window_data); std(window_data); max(window_data); min(window_data)];

            % store features and labels for training
            % flatten into row  
            X_train = [X_train; windo_features(:)'];
            y_train = [y_train;window_label];
        end 
end 

% train random forest 

num_trees = 100;
rfModel= TreeBagger(num_trees,X_train,y_train, 'Method', 'classification');

% load and process test data 
X_test = [];
y_test = [];
time_test = [];
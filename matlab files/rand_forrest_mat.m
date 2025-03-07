% We are going to load in 10csv files 
% 8 of the files we will train and 2 will be for testing 
% They each hve time  for their timestamp 
% The main purpose of this code is to train a random forest model 
% It also evaluates performance on the test data

clc; 
clear all;
close all;
warning off; 

file_names = {
    'trial1_with_time.csv', 'trial2_with_time.csv', 'trial3_with_time.csv', 'trial4_with_time.csv', 'trial5_with_time.csv', ...
    'trial6_with_time.csv', 'trial7_with_time.csv', 'trial8_with_time.csv', 'trial9_with_time.csv', 'trial10_with_time.csv'
};

num_files = length(file_names);
train_files = 8;  % Number of training files
test_files = num_files - train_files;  % Number of test files

% Initialize empty arrays for training and testing
X_train = [];
y_train = [];
X_test = [];
y_test = [];
% Store time values for test dataset
time_test = []; 
% Now im goign to parse through the files and process the data 
for i = 1:num_files
    % now i can load in the csv file 
    data = readtable(file_names{i});

    % here extract the time values. 
    % going to keep it for plotting but not for the training aspect
    time = data.("Time_s_");

    % Now lets define the slip condition 
    % Slip occurs if Sensors 8< 1000 or if any of Sensors 1-7 are > 500
    %  Apply slip detection rule:
    data.Slip = (data.Sensor8 < 1000) | any(table2array(data(:, 1:7)) > 500, 2);
    % Convert logical to numerical (1 = Slip, 0 = No Slip)
    data.Slip = double(data.Slip);

    % now extract the features from all the sensors and he labels (slip)
    X = table2array(data(:,2:9)); % Don't include time 
    y = data.Slip;

    % now split the files into training and testing datasets 

    if i <= train_files
        X_train = [X_train; X]; %append the training data 
        y_train = [y_train; y];
    else 
        X_test = [X_test; X]; % append the test data
        y_test = [y_test; y];
        time_test = [time_test; time]; % davethe time values for plotting 
    end 
end 

% Train the random forest model 
numTrees = 5; % number of decission trees 
fprintf('Training Random Forest with %d trees on %d training files...\n', numTrees, train_files);

% train the random forest model
% NOTE there are different ways to do this trainign lets try this way first 
rfModel = TreeBagger(numTrees,X_train,y_train, "Method","classification");

% Now here I do the test model on the unseen data

fprintf('Testing model on %d test files ...\n',test_files);

% make predictions on the test data
y_pred = str2double(predict(rfModel, X_test));

% calculate the accuracy 
accuracy = sum(y_pred ==y_test)/ length(y_test);
fprintf('Model Accuracy: %.2f%%\n', accuracy*100);

% now lets calculate the confucion matrix 
confMat = confusionmat(y_test,y_pred);
disp('Confision Matrix: ')
disp(confMat)

% Now let's plot actua; VS predicted slip detection 
% figure;
% hold on;
% plot(time_test, y_test, 'bo-', 'MarkerSize', 7, 'DisplayName', 'Actual Slip'); % Blue = Actual
% plot(time_test, y_pred, 'r.-', 'MarkerSize', 10, 'DisplayName', 'Predicted Slip'); % Red = Predicted
% xlabel('Time (s)');
% ylabel('Slip Status (0 = No Slip, 1 = Slip)');
% title('Actual vs Predicted Slip Detection Over Time');
% legend;
% grid on;
% hold off;

figure;
hold on;

% Define light blue color for actual slip regions
light_blue = [0.7, 0.9, 1]; % Light blue RGB color

% Shade background where actual slip occurred (y_test == 1)
area(time_test, y_test, 'FaceColor', light_blue, 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Actual Slip');

% Overlay actual slip values as blue circles
plot(time_test, y_test, 'bo-', 'MarkerSize', 7, 'DisplayName', 'Actual Slip');

% Overlay predicted slip values as red dots
plot(time_test, y_pred, 'r.-', 'MarkerSize', 10, 'DisplayName', 'Predicted Slip');

xlabel('Time (s)');
ylabel('Slip Status (0 = No Slip, 1 = Slip)');
title('Actual vs Predicted Slip Detection Over Time');
legend;
grid on;
hold off;

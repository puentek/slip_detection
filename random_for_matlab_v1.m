% random forrest practuce in matlab


% clear the workspace
clear; clc; close all;

% Here im am going to load in the sensor data from the csv i have 
filename = 'trial1_with_time.csv';  % Update with actual file name
data = readtable(filename);

%  here I will be checking if there is a time column if not then I'll generate the sampling rate
sampling_rate = 10; % this value is in Hz
if contains(data.Properties.VariableNames{1}, 'Sensor')
    time = (0:height(data)-1)' / sampling_rate;
    data.Time = time;
end 

% here I will assigning labels to the data loaded in 
% and extract the information I need from the csv file

num_sensors = width(data)-1;  % don't want to use the time column right now 
features = data(:,1:num_sensors); % now i'm pulling out those values 

% I will have a slip condition set here to track when it is slipping or not. 
% This is a placeholder to be  edited after colecting ground truth data 

% slip occurs if sensor 8 < 1000 ot if any sesnors 1-7 are > 500
labels = (data.Sensor8 < 1000) | any(table2array(data(:, 1:7)) > 500,2);

% now i'm converting the table to a numerical array so i can do some training
X = table2array(features); % make a matrix for the sensor readings 
y = labels; % this is the target variable of slip = 1, no slip = 0

%  and now wohoo we get to do this tarining after we split some data. 
% 80% training of the graphs we made before and testing just on 20% of the data 
cv = cvpartition(size(X,1), 'HoldOut', 0.2); % 80% training , 20% testing 
X_train = X(training(cv),:); % this is the training features 
y_train = y(training(c),:); % Training labels 
X_test = X(test(cv), :); % Testing features 
y_test = y(test(cv, :)); %testing labels


% lets get to training!!
numTrees = 5; % this is the number of decision trees in the forest 
fprintf('Training Random Forest with %d trees ...\n',numTrees);

rfModel = TreeBagger(numTrees, X_train, y_train, 'Method', 'classification');


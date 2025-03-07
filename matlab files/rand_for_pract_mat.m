clc; 
clear all;
close all;
warning off; 
data = readtable('study_data.csv');
% creates an array which stores distinct types
k = ["High", "Low"];

% Here we want to encode high to 1 and low to 0
l = [1, 0];
g = data.knowledge_level;
number = zeros(length(g),1);
% this section will do that replacement for the csv file 
for i = 1:length(k)
    rs = ismember(g,k(i));
    number(rs) = l(i);
end 
% now I want to create a new column category 
data.category_encoded = number; 
% drop knowledge level bc we don't need that anymore 
data.knowledge_level = [];
% here we use the holdout method and keeping 30% for testing purpose 
cv = cvpartition(size(data,1), 'HoldOut', 0.3);
% now extract the index
idx = cv.test;
dataTrain = data(~idx,:);
dataTest = data(idx,:);
testing = dataTest(1:end,1:end-1);

% this is the classification tree %UNCOMMENT IF NEEDED 
% model=fitctree(dataTrain,'category_encoded');

% now we want to apply a random forrest aspect to it 
model=fitensemble(dataTrain,'category_encoded','Bag',100,'Tree','Type','classification')
% category_encoded is the column with the output parameter
% 'Tree makes this a a random forrest classification type 
prediction = predict(model,testing);
% here I am getting the accuray 
ms = (sum(prediction==table2array(dataTest(:,end)))/size(dataTest,1))*100
e=min(data.x___repetition_time):0.01:max(data.x___repetition_time);
f=min(data.study_time):0.01:max(data.study_time);
[x1 x2]=meshgrid(e,f);
x=[x1(:) x2(:)];
ms=predict(model,x);
gscatter(x1(:),x2(:),ms,'cy');
hold on;
gscatter(dataTrain.x___repetition_time,dataTrain.study_time,dataTrain.category_encoded,'rg','.',30);
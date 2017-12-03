%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Machine Learning Project Assignment
%   Comparison of Classifiers
% 
%   Optimization and Data Analytics (E17)
%   Aarhus University
%
%   Hannes Bartle
%   hannes.bartle@uni.au.dk
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc

%% Set Path Variable
disp('Setting path...')
addpath('functions/')
addpath('scripts/')
addpath('classifiers/')
addpath(genpath('data/'))
disp('Done!')
%% Load the MNIST Image Data
disp('Loading MNIST data Set...')
train_images_MNIST = loadMNISTImages('train-images.idx3-ubyte');
test_images_MNIST = loadMNISTImages('t10k-images.idx3-ubyte');
train_labels_MNIST = loadMNISTLabels('train-labels.idx1-ubyte');
test_labels_MNIST = loadMNISTLabels('t10k-labels.idx1-ubyte');
disp('Done!')
%% Load the ORL Image Data
disp('Loading ORL data set...')
load('orl_data.mat')
load('orl_lbls.mat')
images_ORL = data;
clear data
labels_ORL = lbls;
clear lbls

% Split up ORL Data
[~,number_of_images]= size(images_ORL);

train_percentage = 0.8;
train_number = 0.8*number_of_images;
test_number = number_of_images - train_number;

% Random Permutation
perm = randperm(number_of_images);

% Shuffle Images
train_images_ORL = images_ORL(:,perm(1:train_number));
test_images_ORL = images_ORL(:,perm(train_number+1:end));
% Shuffle the corresponding labels
train_labels_ORL = labels_ORL(perm(1:train_number));
test_labels_ORL = labels_ORL(perm(train_number+1:end));

disp('Done!')

%% Dimensionality Reduction using PCA
do_PCA = false;
if do_PCA == true
    apply_PCA
end
%% Nearest Centroid Classifier Evaluation
apply_NCC
%% Nearest Subclass Classifier
apply_NSC
%% Nearest Neighborhood Classifier
apply_NN
%% Perceptron with Backpropagation on original Data
apply_Perceptron_BP
%% Perceptron Minimum Square Error
apply_Perceptron_MSE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Scoring
score
%% Plots
FontSize= 16;
fig_MNIST = figure('units','normalized','outerposition',[0 0 1 1]);
y = [sc_nc_MNIST;...
     sc_nsc_MNIST(1);...
     sc_nsc_MNIST(2);...
     sc_nsc_MNIST(3);...
     sc_nn_MNIST;...
     sc_pbp_MNIST;...
     sc_pmse_MNIST]*100;
c = {'NC','NSC2','NSC3','NSC5','NN','P-BP','P-MSE'};
bar(y)
barvalues;
ylim([0 100]);
set(gca,'xticklabel',c)
set(gca,'FontSize',FontSize);
ylabel('Success Rate [%]','FontSize',FontSize)
grid on


fig_ORL = figure('units','normalized','outerposition',[0 0 1 1]);
y = [sc_nc_MNIST;...
     sc_nsc_ORL(1);...
     sc_nsc_ORL(2);...
     sc_nsc_ORL(3);...
     sc_nn_ORL;...
     sc_pbp_ORL;...
     sc_pmse_ORL]*100;
c = {'NC','NSC2','NSC3','NSC5','NN','P-BP','P-MSE'};
bar(y)
barvalues;
ylim([0 100]);
set(gca,'xticklabel',c)
set(gca,'FontSize',FontSize);
ylabel('Success Rate [%]','FontSize',FontSize)
grid on








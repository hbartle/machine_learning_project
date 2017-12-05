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

train_percentage = 0.7;
train_number = 0.7*number_of_images;
test_number = number_of_images - train_number;

% Random Permutation
perm = randperm(number_of_images);

% % Shuffle Images
% train_images_ORL = images_ORL(:,perm(1:train_number));
% test_images_ORL = images_ORL(:,perm(train_number+1:end));
% % Shuffle the corresponding labels
% train_labels_ORL = labels_ORL(perm(1:train_number));
% test_labels_ORL = labels_ORL(perm(train_number+1:end));

train_images_ORL = [];
test_images_ORL =[];
train_labels_ORL = [];
test_labels_ORL = [];
for i=1:40
    train_images_ORL = [train_images_ORL images_ORL(:,find(labels_ORL==i,7))];
    test_images_ORL = [test_images_ORL images_ORL(:,find(labels_ORL==i,3,'last'))];
    train_labels_ORL = [train_labels_ORL labels_ORL(find(labels_ORL==i,7))];
    test_labels_ORL = [test_labels_ORL labels_ORL(find(labels_ORL==i,3))];
end
disp('Done!')

%% Dimensionality Reduction using PCA
do_PCA = true;
target_dimension = 1:10;
if do_PCA == true
    apply_PCA
end
%% Nearest Centroid Classifier Evaluation
apply_NCC
%% Nearest Subclass Classifier
subclasses = [2,3,5];
apply_NSC
%% Nearest Neighborhood Classifier
training_subset = 1000;
testing_subset = 500;
apply_NN
%% Perceptron with Backpropagation on original Data
apply_Perceptron_BP
%% Perceptron Minimum Square Error
apply_Perceptron_MSE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Scoring
score
%% Plots
close all
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
ylim([0 120]);
set(gca,'xticklabel',c)
set(gca,'FontSize',FontSize);
ylabel('Success Rate [%]','FontSize',FontSize)
grid on


fig_ORL = figure('units','normalized','outerposition',[0 0 1 1]);
y = [sc_nc_ORL;...
     sc_nsc_ORL(1);...
     sc_nsc_ORL(2);...
     sc_nsc_ORL(3);...
     sc_nn_ORL;...
     sc_pbp_ORL;...
     sc_pmse_ORL]*100;
c = {'NC','NSC2','NSC3','NSC5','NN','P-BP','P-MSE'};
bar(y)
barvalues;
ylim([0 120]);
set(gca,'xticklabel',c)
set(gca,'FontSize',FontSize);
ylabel('Success Rate [%]','FontSize',FontSize)
grid on

fig_MNIST_time = figure('units','normalized','outerposition',[0 0 1 1]);
y = [t_nc_MNIST;...
     t_nsc_MNIST{1};...
     t_nsc_MNIST{2};...
     t_nsc_MNIST{3};...
     t_nn_MNIST;...
     t_pbp_MNIST;...
     t_pmse_MNIST]*1000;
c = {'NC','NSC2','NSC3','NSC5','NN','P-BP','P-MSE'};
bar(y)
barvalues;
set(gca,'xticklabel',c)
set(gca,'FontSize',FontSize);
ylabel('Execution Time [ms]','FontSize',FontSize)
grid on

fig_ORL_time = figure('units','normalized','outerposition',[0 0 1 1]);
y = [t_nc_ORL;...
     t_nsc_ORL{1};...
     t_nsc_ORL{2};...
     t_nsc_ORL{3};...
     t_nn_ORL;...
     t_pbp_ORL;...
     t_pmse_ORL]*1000;
c = {'NC','NSC2','NSC3','NSC5','NN','P-BP','P-MSE'};
bar(y)
barvalues;
set(gca,'xticklabel',c)
set(gca,'FontSize',FontSize);
ylabel('Execution Time [ms]','FontSize',FontSize)
grid on







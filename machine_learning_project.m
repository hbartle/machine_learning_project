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
addpath('misc/')
addpath('classifiers/')
addpath(genpath('data/'))

%% Load the MNIST Image Data
train_images_MNIST = loadMNISTImages('train-images.idx3-ubyte');
test_images_MNIST = loadMNISTImages('t10k-images.idx3-ubyte');
train_labels_MNIST = loadMNISTLabels('train-labels.idx1-ubyte');
test_labels_MNIST = loadMNISTLabels('t10k-labels.idx1-ubyte');

%% Load the ORL Image Data
load('orl_data.mat')
load('orl_lbls.mat')
images_ORL = data;
clear data
labels_ORL = lbls;
clear lbls

%% Split up ORL Data
[~,number_of_images]= size(images_ORL);

train_percentage = 0.7;
train_number = 0.7*number_of_images;
test_number = number_of_images - train_number;

% Random Permutation
perm = randperm(number_of_images);

% Shuffle Images
train_images_ORL = images_ORL(:,perm(1:train_number));
test_images_ORL = images_ORL(:,perm(train_number+1:end));
% Shuffle the corresponding labels
train_labels_ORL = labels_ORL(perm(1:train_number));
test_labels_ORL = labels_ORL(perm(train_number+1:end));


%% Dimensionality Reduction using PCA
target_dimension = 1;

train_images_MNIST_pca = principalComponents(train_images_MNIST,...
                                             target_dimension);
test_images_MNIST_pca = principalComponents(test_images_MNIST,...
                                            target_dimension);
train_images_ORL_pca = principalComponents(train_images_ORL,...
                                           target_dimension);
test_images_ORL_pca = principalComponents(test_images_ORL,...
                                          target_dimension);


%% Nearest Class Classifier Evaluation
tic;
% Classification on Raw Image Data
nc_labels_MNIST = ncClassifier(train_images_MNIST(:,:),...
                               test_images_MNIST(:,:),...
                               train_labels_MNIST(:),...
                               'MNIST');
t_nc_MNIST = toc;

tic;
nc_labels_ORL = ncClassifier(train_images_ORL,...
                             test_images_ORL,...
                             train_labels_ORL,...
                             'ORL');
t_nc_ORL = toc;

% Classification on PCA-reduced Image Data
tic;
nc_labels_MNIST_pca = ncClassifier(train_images_MNIST_pca,...
                                                  test_images_MNIST_pca,...
                                                  train_labels_MNIST,...
                                                  'MNIST');
t_nc_MNIST_pca = toc;
tic;
nc_labels_ORL_pca = ncClassifier(train_images_ORL_pca,...
                                              test_images_ORL_pca,...
                                              train_labels_ORL,...
                                              'ORL');
t_nc_ORL_pca = toc;
%% Nearest Subclass Classifier

% Classification on Raw Image Data
tic;
nsc_labels_MNIST = nscClassifier(train_images_MNIST,...
                               test_images_MNIST,...
                               train_labels_MNIST,...
                               5,...
                               'MNIST');
t_nsc_MNIST = toc;
tic;
nsc_labels_ORL = nscClassifier(train_images_ORL,...
                               test_images_ORL,...
                               train_labels_ORL,...
                               5,...
                               'ORL');
t_nsc_ORL = toc;

% Classification on PCA-reduced Image Data
tic;
nsc_labels_MNIST_pca = nscClassifier(train_images_MNIST_pca,...
                               test_images_MNIST_pca,...
                               train_labels_MNIST,...
                               2,...
                               'MNIST');
t_nsc_MNIST_pca = toc;
tic;
nsc_labels_ORL_pca = nscClassifier(train_images_ORL_pca,...
                               test_images_ORL_pca,...
                               train_labels_ORL,...
                               2,...
                               'ORL');
t_nsc_ORL_pca = toc;

%% Nearest Neighborhood Classifier

% Classification on Raw Image Data
training_subset = 700;
testing_subset = 300;
tic;
nn_labels_MNIST = nnClassifier(train_images_MNIST(:,1:training_subset),...
                               test_images_MNIST(:,1:testing_subset),...
                               train_labels_MNIST(1:training_subset));
t_nn_MNIST = toc;
tic;
nn_labels_ORL = nnClassifier(train_images_ORL,...
                               test_images_ORL,...
                               train_labels_ORL);
t_nn_ORL = toc;

% Classification on PCA-reduced Image Data
tic;
nn_labels_MNIST_pca = nnClassifier(train_images_MNIST_pca(:,1:training_subset),...
                               test_images_MNIST_pca(:,1:testing_subset),...
                               train_labels_MNIST(1:training_subset));
t_nn_MNIST_pca = toc;
tic;
nn_labels_ORL_pca = nnClassifier(train_images_ORL_pca,...
                               test_images_ORL_pca,...
                               train_labels_ORL);
t_nn_ORL_pca = toc;
%% Perceptron with Backpropagation on original Data
tic;
W_MNIST = trainPerceptronBP(train_images_MNIST,...
                      train_labels_MNIST,...
                      0.1,...
                      'MNIST');           
pBP_labels_MNIST = perceptronBP(W_MNIST,test_images_MNIST,'MNIST');
t_pbp_MNIST = toc;
tic;                  
W_ORL = trainPerceptronBP(train_images_ORL,...
                      train_labels_ORL,...
                      0.1,...
                      'ORL'); 
pBP_labels_ORL = perceptronBP(W_ORL,test_images_ORL,'ORL');
t_pbp_ORL = toc;

% Perceptron with Backpropagation on PCA-reduced Data
tic;
W_MNIST_pca = trainPerceptronBP(train_images_MNIST_pca,...
                      train_labels_MNIST,...
                      0.1,...
                      'MNIST');              
pBP_labels_MNIST_pca = perceptronBP(W_MNIST_pca,test_images_MNIST_pca,'MNIST');
t_pbp_MNIST_pca = toc;                               
tic;
W_ORL_pca = trainPerceptronBP(train_images_ORL_pca,...
                      train_labels_ORL,...
                      0.1,...
                      'ORL'); 
pBP_labels_ORL_pca = perceptronBP(W_ORL_pca,test_images_ORL_pca,'ORL');
t_pbp_ORL_pca = toc;

%% Perceptron Minimum Square Error

% Original Data
tic;
W_MNIST_MSE = trainPerceptronMSE(train_images_MNIST,...
                              train_labels_MNIST,'MNIST'); 
pMSE_labels_MNIST = perceptronMSE(W_MNIST_MSE,test_images_MNIST,'MNIST');
t_pmse_MNIST = toc;
tic;
W_ORL_MSE = trainPerceptronMSE(train_images_ORL,...
                              train_labels_ORL,'ORL'); 
pMSE_labels_ORL = perceptronMSE(W_ORL_MSE,test_images_ORL,'ORL');
t_pmse_ORL = toc;

% PCA-reduced Data
tic;
W_MNIST_MSE_pca = trainPerceptronMSE(train_images_MNIST_pca,...
                              train_labels_MNIST,'MNIST'); 
pMSE_labels_MNIST_pca = perceptronMSE(W_MNIST_MSE_pca,...
                                      test_images_MNIST_pca,'MNIST');
t_pmse_MNIST_pca = toc;
tic;
W_ORL_MSE_pca = trainPerceptronMSE(train_images_ORL_pca,...
                              train_labels_ORL,'ORL'); 
pMSE_labels_ORL_pca = perceptronMSE(W_ORL_MSE_pca,test_images_ORL_pca,'ORL');
t_pmse_ORL_pca = toc;     

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Scoring

% Nearest Centroid
sc_nc_MNIST = scoreClassifier(nc_labels_MNIST,test_labels_MNIST(:));
sc_nc_ORL   = scoreClassifier(nc_labels_ORL,test_labels_ORL);
sc_nc_MNIST_pca = scoreClassifier(nc_labels_MNIST_pca,test_labels_MNIST);
sc_nc_ORL_pca   = scoreClassifier(nc_labels_ORL_pca,test_labels_ORL);

% Nearest Subclass
sc_nsc_MNIST = scoreClassifier(nsc_labels_MNIST,test_labels_MNIST);
sc_nsc_ORL   = scoreClassifier(nsc_labels_ORL,test_labels_ORL);
sc_nsc_MNIST_pca = scoreClassifier(nsc_labels_MNIST_pca,test_labels_MNIST);
sc_nsc_ORL_pca   = scoreClassifier(nsc_labels_ORL_pca,test_labels_ORL);

% Nearest Neighbor
sc_nn_MNIST = scoreClassifier(nn_labels_MNIST,test_labels_MNIST(1:testing_subset));
sc_nn_ORL   = scoreClassifier(nn_labels_ORL,test_labels_ORL);
sc_nn_MNIST_pca = scoreClassifier(nn_labels_MNIST_pca,test_labels_MNIST(1:testing_subset));
sc_nn_ORL_pca   = scoreClassifier(nn_labels_ORL_pca,test_labels_ORL);

% BP Perceptron
sc_pbp_MNIST = scoreClassifier(pBP_labels_MNIST,test_labels_MNIST);
sc_pbp_ORL   = scoreClassifier(pBP_labels_ORL,test_labels_ORL);
sc_pbp_MNIST_pca = scoreClassifier(pBP_labels_MNIST_pca,test_labels_MNIST);
sc_pbp_ORL_pca   = scoreClassifier(pBP_labels_ORL_pca,test_labels_ORL);

% MSE Perceptron
sc_pmse_MNIST = scoreClassifier(pMSE_labels_MNIST,test_labels_MNIST);
sc_pmse_ORL   = scoreClassifier(pMSE_labels_ORL,test_labels_ORL);
sc_pmse_MNIST_pca = scoreClassifier(pMSE_labels_MNIST_pca,test_labels_MNIST);
sc_pmse_ORL_pca   = scoreClassifier(pMSE_labels_ORL_pca,test_labels_ORL);

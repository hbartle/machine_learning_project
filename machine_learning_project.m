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


% Classification on Raw Image Data
nc_labels_MNIST = ncClassifier(train_images_MNIST(:,:),...
                               test_images_MNIST(:,:),...
                               train_labels_MNIST(:),...
                               'MNIST');

nc_labels_ORL = ncClassifier(train_images_ORL,...
                             test_images_ORL,...
                             train_labels_ORL,...
                             'ORL');



scoreClassifier(nc_labels_MNIST,test_labels_MNIST(:))
scoreClassifier(nc_labels_ORL,test_labels_ORL)

%% Classification on PCA-reduced Image Data
nc_labels_MNIST_pca = ncClassifier(train_images_MNIST_pca,...
                                                  test_images_MNIST_pca,...
                                                  train_labels_MNIST,...
                                                  'MNIST');

nc_labels_ORL_pca = ncClassifier(train_images_ORL_pca,...
                                              test_images_ORL_pca,...
                                              train_labels_ORL,...
                                              'ORL');
                          
                          

scoreClassifier(nc_labels_MNIST_pca,test_labels_MNIST)
scoreClassifier(nc_labels_ORL_pca,test_labels_ORL)
                         

%% Nearest Subclass Classifier

% Classification on Raw Image Data
nsc_labels_MNIST = nscClassifier(train_images_MNIST,...
                               test_images_MNIST,...
                               train_labels_MNIST,...
                               5,...
                               'MNIST');
nsc_labels_ORL = nscClassifier(train_images_ORL,...
                               test_images_ORL,...
                               train_labels_ORL,...
                               5,...
                               'ORL');

scoreClassifier(nsc_labels_MNIST,test_labels_MNIST)
scoreClassifier(nsc_labels_ORL,test_labels_ORL)

%% Classification on PCA-reduced Image Data
nsc_labels_MNIST_pca = nscClassifier(train_images_MNIST_pca,...
                               test_images_MNIST_pca,...
                               train_labels_MNIST,...
                               2,...
                               'MNIST');
nsc_labels_ORL_pca = nscClassifier(train_images_ORL_pca,...
                               test_images_ORL_pca,...
                               train_labels_ORL,...
                               2,...
                               'ORL');

scoreClassifier(nsc_labels_MNIST_pca,test_labels_MNIST)
scoreClassifier(nsc_labels_ORL_pca,test_labels_ORL)


%% Nearest Neighborhood Classifier

% Classification on Raw Image Data
training_subset = 700;
testing_subset = 300;
nn_labels_MNIST = nnClassifier(train_images_MNIST(:,1:training_subset),...
                               test_images_MNIST(:,1:testing_subset),...
                               train_labels_MNIST(1:training_subset));
scoreClassifier(nn_labels_MNIST,test_labels_MNIST(1:testing_subset))
  
nn_labels_ORL = nnClassifier(train_images_ORL,...
                               test_images_ORL,...
                               train_labels_ORL);
scoreClassifier(nn_labels_ORL,test_labels_ORL)
%%
% Classification on PCA-reduced Image Data
nn_labels_MNIST_pca = nnClassifier(train_images_MNIST_pca(:,1:training_subset),...
                               test_images_MNIST_pca(:,1:testing_subset),...
                               train_labels_MNIST(1:training_subset));
scoreClassifier(nn_labels_MNIST_pca,test_labels_MNIST(1:testing_subset))

nn_labels_ORL_pca = nnClassifier(train_images_ORL_pca,...
                               test_images_ORL_pca,...
                               train_labels_ORL);
scoreClassifier(nn_labels_ORL_pca,test_labels_ORL)


%% Perceptron with Backpropagation on original Data

W_MNIST = trainPerceptronBP(train_images_MNIST,...
                      train_labels_MNIST,...
                      0.1,...
                      'MNIST');
                  
                  
pBP_labels_MNIST = perceptronBP(W_MNIST,test_images_MNIST,'MNIST');
                  
scoreClassifier(pBP_labels_MNIST,test_labels_MNIST)
                 
      
W_ORL = trainPerceptronBP(train_images_ORL,...
                      train_labels_ORL,...
                      0.1,...
                      'ORL'); 
pBP_labels_ORL = perceptronBP(W_ORL,test_images_ORL,'ORL');

scoreClassifier(pBP_labels_ORL,test_labels_ORL)

%% Perceptron with Backpropagation on PCA-reduced Data

W_MNIST_pca = trainPerceptronBP(train_images_MNIST_pca,...
                      train_labels_MNIST,...
                      0.1,...
                      'MNIST');
                  
pBP_labels_MNIST_pca = perceptronBP(W_MNIST_pca,test_images_MNIST_pca,'MNIST');
                  
scoreClassifier(pBP_labels_MNIST_pca,test_labels_MNIST)
                 
   
W_ORL_pca = trainPerceptronBP(train_images_ORL_pca,...
                      train_labels_ORL,...
                      0.1,...
                      'ORL'); 
pBP_labels_ORL_pca = perceptronBP(W_ORL_pca,test_images_ORL_pca,'ORL');

scoreClassifier(pBP_labels_ORL_pca,test_labels_ORL)

%% Perceptron Minimum Square Error

% Original Data
W_MNIST_MSE = trainPerceptronMSE(train_images_MNIST,...
                              train_labels_MNIST,'MNIST'); 
pMSE_labels_MNIST = perceptronMSE(W_MNIST_MSE,test_images_MNIST,'MNIST');

scoreClassifier(pMSE_labels_MNIST,test_labels_MNIST)

W_ORL_MSE = trainPerceptronMSE(train_images_ORL,...
                              train_labels_ORL,'ORL'); 
pMSE_labels_ORL = perceptronMSE(W_ORL_MSE,test_images_ORL,'ORL');

scoreClassifier(pMSE_labels_ORL,test_labels_ORL)

%% PCA-reduced Data
W_MNIST_MSE_pca = trainPerceptronMSE(train_images_MNIST_pca,...
                              train_labels_MNIST,'MNIST'); 
pMSE_labels_MNIST_pca = perceptronMSE(W_MNIST_MSE_pca,...
                                      test_images_MNIST_pca,'MNIST');

scoreClassifier(pMSE_labels_MNIST_pca,test_labels_MNIST)

W_ORL_MSE_pca = trainPerceptronMSE(train_images_ORL_pca,...
                              train_labels_ORL,'ORL'); 
pMSE_labels_ORL_pca = perceptronMSE(W_ORL_MSE_pca,test_images_ORL_pca,'ORL');

scoreClassifier(pMSE_labels_ORL_pca,test_labels_ORL)

         



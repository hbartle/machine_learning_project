function [ W ] = trainPerceptronMSE(data_train,labels_train,data_type)
%TRAIN_PERCEPTRON_MSE Train Perceptron using Minimal Square Error Algorithm

if strcmp(data_type,'MNIST')
    % MNIST Data
    % 10 Classes
    number_of_classes = 10;
    label_correction=1;
elseif strcmp(data_type,'ORL')
    % ORL Data
    % 40 Classes
    number_of_classes = 40;
    label_correction=0;
end
[dim,number_of_samples] = size(data_train);

% Get the desired output vector
l = -1 * ones(number_of_classes,number_of_samples);
for i=1:number_of_samples
    l(labels_train(i)+label_correction,i) = 1;
end


% Get the pseudo-inverse of the transpose training data 
X_p = (data_train*data_train' + 0.001*eye(dim))^-1*data_train;

% Get the Weights
W = X_p*l';



end


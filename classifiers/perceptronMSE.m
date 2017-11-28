function [ labels ] = perceptronMSE(W,data_test,data_type)
%PERCEPTRON_MSE Minimum Square Error Perceptron Classifier

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


[~,number_of_samples] = size(data_test);

labels = nan*ones(1,number_of_samples);

% Classify the Samples
g =  W'*data_test;

for i=1:number_of_classes
    k = find(g(i,:) > 0);
    labels(k) = (i-label_correction)*ones(1,length(k));
end



end


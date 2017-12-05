function [ W ] = trainPerceptronBP(data_train,labels_train,learning_rate,data_type )
%TRAIN_PERCEPTRON_BP Train a perceptron using Backpropagation


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

% Augment the data
data = [ data_train; ones(1,number_of_samples)];

% Initialize weights
W = 0.1*[ones(dim,number_of_classes);zeros(1,number_of_classes)];

% Get the Binary Labels
l = -1 * ones(number_of_classes,number_of_samples);
for i=1:number_of_samples
    l(labels_train(i)+ label_correction,i) = 1;
end

% Set of all misclassified samples
X = 0;
% Perceptron Criterion Function
f = zeros(number_of_classes,number_of_samples);

% Counter to stop iteration when not finding a perfect solution
counter = 0;
while ~isempty(X) & counter < 200
    % Calculate the Perceptron Criterion Function
    for i=1:number_of_samples
        f(:,i) =  l(:,i).*(W'*data(:,i));
    end
    
    for i=1:number_of_classes
        % Find all misclassified samples
        X = find(f(i,:) < 0);
        
        % Calculate Gradient
        delta_W = learning_rate* sum((ones(dim+1,1)*l(i,X)).*data(:,X),2);
        W(:,i) = W(:,i) + delta_W;
    end

    counter = counter + 1;
end
length(X)
counter
end


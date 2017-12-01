function [ labels,m ] = ncClassifier(data_train, data_test, labels_train, data_type)
%NC_CLASSIFY Nearest Centroid Classifier


if strcmp(data_type,'MNIST')  
    % MNIST Data
    % 0-9 Numbers resulting in 10 classes
    number_of_classes= 10;
    label_correction = 1;
elseif strcmp(data_type,'ORL') 
    % ORL Data
    % 40 Classes
    number_of_classes= 40;
    label_correction = 0;  
end

% Calculate Class Mean
for i=1:number_of_classes
    idx = find(labels_train == i-label_correction);
    m(:,i)= mean(data_train(:,idx),2);
end


% Classify the test images
[~,number_of_samples] = size(data_test);
for i=1:number_of_samples
    for k =1:number_of_classes
        d(k) = norm(data_test(:,i) -m(:,k))^2;
    end
    [~,labels(i)] = min(d) ; 
    labels(i) = labels(i) - label_correction;
end


end


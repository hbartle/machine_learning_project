function [ labels ] = nscClassifier(data_train,data_test,labels_train,K,data_type)
%NSC_CLASSIFY Nearest Subclass Centroid Classifier

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

subclass_mean = cell(1,number_of_classes);
% Calculate Subclasses 
for i=1:number_of_classes
    idx = find(labels_train == i-label_correction);
    [~,subclass_mean{i}] = kmeans(data_train(:,idx)',K);
end


% Classify the test images
[~,number_of_samples] = size(data_test);
for i=1:number_of_samples
    for k =1:number_of_classes
        for j = 1:K
            d_sub(j) = norm(data_test(:,i) - subclass_mean{k}(j,:)')^2;
        end
        d(k) = min(d_sub);
    end
    [~,c] = min(d);
    labels(i) = c - label_correction;
end


end


function [ labels ] = nnClassifier(data_train,data_test, labels_train)
%NN_CLASSIFY Nearest Neighborhood Classifier

% Classify the test images
[~,number_of_samples] = size(data_test);
[~,number_of_training_samples] = size(data_train);
for i=1:number_of_samples
    for k =1:number_of_training_samples
        d(k) = norm(data_test(:,i) - data_train(:,k))^2;
    end
    [~,labels(i)] = min(d);
    labels(i) = labels_train(labels(i));
end

end



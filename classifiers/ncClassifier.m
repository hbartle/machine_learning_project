function [ lables,m ] = ncClassifier(data_train, data_test, labels_train, data_type)
%NC_CLASSIFY Nearest Centroid Classifier


if strcmp(data_type,'MNIST')  
    % MNIST Data
    % 0-9 Numbers resulting in 10 classes
    
    % Calculate Class Mean
    for i=0:9
        idx = find(labels_train == i);
        m(:,i+1)= mean(data_train(:,idx),2);
    end
    
    
    % Classify the test images
    [~,number_of_samples] = size(data_test);
    for i=1:number_of_samples
        for k =1:10
            d(k) = norm(data_test(:,i) -m(:,k))^2;
        end
        [~,lables(i)] = min(d);  
    end
  
elseif strcmp(data_type,'ORL') 
    % ORL Data
    % 40 people -> 40 classes
    
    % Calculate Class Mean
    for i=1:40
        idx = find(labels_train == i);
        m(:,i)= mean(data_train(:,idx),2);
    end
    
    % Classify the test images
    [~,number_of_samples] = size(data_test);
    for i=1:number_of_samples
        for k =1:40
            d(k) = norm(data_test(:,i) -m(:,k))^2;
        end
        [~,lables(i)] = min(d);  
    end
   
    
end

end


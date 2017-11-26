function [ lables,m ] = nc_classify(data_train, data_test, labels_train )
%NC_CLASSIFY Nearest Centroid Classifier

s = size(data_train);

if s(1) == 784 
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
        [~,lables(i)] = max(d);  
    end
  
elseif s(1) == 1200
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
        [~,lables(i)] = max(d);  
    end
   
    
end

end


function [ labels,centroid] = kMeans(data, K, eta )
%KMEANS Cluster Data into K Clusters
if nargin <3
    eta = 1e-5;
end


[~,number_of_samples] = size(data);

% Random Initial Centroids
centroid = data(:,randi(number_of_samples,1,K));
last_centroid = zeros(size(centroid));

d = zeros(1,number_of_samples);
labels = zeros(1,number_of_samples);


% K-Means Clustering
while norm(centroid - last_centroid) > eta  
    last_centroid = centroid;
    for i=1:number_of_samples
        for l=1:K
            d(l) = norm(data(:,i) - centroid(:,l))^2;
        end
        [~, labels(i)] = min(d);
    end
    % Update Centroids
    for l=1:K
        centroid(:,l) = mean(data(:,labels==l),2);
    end
end


end


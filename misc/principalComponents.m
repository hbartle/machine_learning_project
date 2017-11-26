function [ transformed_samples] = principalComponents( samples,target_dimension)
%PRINCIPALCOMPONENTS Reduce Data Dimension using PCA


[img_size,number_of_samples] = size(samples);
transformed_samples = zeros(target_dimension,number_of_samples);

m = mean(samples,2); 

X = (samples-m*ones(1,number_of_samples));
scatterMatrix = X*X';
[W,D] = eig(scatterMatrix);

% Sort according to biggest Eigenvalue
[~,permutation]=sort(diag(D));
W = W(:,permutation);

% Reduce Dimensions
W = W(:,end-target_dimension+1:end);

% Transform Samples
for i=1:number_of_samples 
    transformed_samples(:,i) = W'*samples(:,i);
end

end


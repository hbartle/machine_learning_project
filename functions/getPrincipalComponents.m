function [ W,m] = getPrincipalComponents( samples,target_dimension)
%PRINCIPALCOMPONENTS Reduce Data Dimension using PCA


[~,number_of_samples] = size(samples);
%transformed_samples = zeros(target_dimension,number_of_samples);

if nargin < 3
    m = mean(samples,2); 
end
X = (samples-m*ones(1,number_of_samples))';
scatterMatrix = cov(X);
[W,D] = eig(scatterMatrix);

% Sort according to biggest Eigenvalue
[~,permutation]=sort(diag(D),'descend');
W = W(:,permutation);
W = W(:,1:target_dimension);

end


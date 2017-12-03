disp('Scoring Classifiers...')

% Original Data
% Nearest Centroid
sc_nc_MNIST = scoreClassifier(nc_labels_MNIST,test_labels_MNIST(:));
sc_nc_ORL   = scoreClassifier(nc_labels_ORL,test_labels_ORL);
% Nearest Subclass
for i = 1:length(subclasses)
    sc_nsc_MNIST(i) = scoreClassifier(nsc_labels_MNIST{i},test_labels_MNIST);
    sc_nsc_ORL(i)   = scoreClassifier(nsc_labels_ORL{i},test_labels_ORL);
end
% Nearest Neighbor
sc_nn_MNIST = scoreClassifier(nn_labels_MNIST,test_labels_MNIST(1:testing_subset));
sc_nn_ORL   = scoreClassifier(nn_labels_ORL,test_labels_ORL);
% BP Perceptron
sc_pbp_MNIST = scoreClassifier(pBP_labels_MNIST,test_labels_MNIST);
sc_pbp_ORL   = scoreClassifier(pBP_labels_ORL,test_labels_ORL);
% MSE Perceptron
sc_pmse_MNIST = scoreClassifier(pMSE_labels_MNIST,test_labels_MNIST);
sc_pmse_ORL   = scoreClassifier(pMSE_labels_ORL,test_labels_ORL);

% PCA-reduced Data
for i = 1:length(target_dimension)
    % Nearest Centroid
    sc_nc_MNIST_pca(i) = scoreClassifier(nc_labels_MNIST_pca{i},test_labels_MNIST);
    sc_nc_ORL_pca(i)   = scoreClassifier(nc_labels_ORL_pca{i},test_labels_ORL);
    % Nearest Subclass
    for k = 1:length(subclasses)
        sc_nsc_MNIST_pca(i,k) = scoreClassifier(nsc_labels_MNIST_pca{k,i},test_labels_MNIST);
        sc_nsc_ORL_pca(i,k)   = scoreClassifier(nsc_labels_ORL_pca{k,i},test_labels_ORL);
    end
    % Nearest Neighbor
    sc_nn_MNIST_pca(i) = scoreClassifier(nn_labels_MNIST_pca{i},test_labels_MNIST(1:testing_subset));
    sc_nn_ORL_pca(i)   = scoreClassifier(nn_labels_ORL_pca{i},test_labels_ORL);
    % BP Perceptron
    sc_pbp_MNIST_pca(i) = scoreClassifier(pBP_labels_MNIST_pca{i},test_labels_MNIST);
    sc_pbp_ORL_pca(i)   = scoreClassifier(pBP_labels_ORL_pca{i},test_labels_ORL);
    % MSE Perceptron
    sc_pmse_MNIST_pca(i) = scoreClassifier(pMSE_labels_MNIST_pca{i},test_labels_MNIST);
    sc_pmse_ORL_pca(i)   = scoreClassifier(pMSE_labels_ORL_pca{i},test_labels_ORL);
end

disp('Done!')

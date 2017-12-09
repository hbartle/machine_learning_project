disp('Scoring Classifiers...')

% Original Data
% Nearest Centroid
try
    sc_nc_MNIST = scoreClassifier(nc_labels_MNIST,test_labels_MNIST(:));
    for k =1:number_of_ORL_iterations
        sc_nc_ORL(k)   = scoreClassifier(nc_labels_ORL{k},test_labels_ORL{k});
    end
catch
    disp('Labels not available')
end
%% Nearest Subclass
try
    for i = 1:length(subclasses)
        sc_nsc_MNIST(i) = scoreClassifier(nsc_labels_MNIST{i},test_labels_MNIST);
        for k =1:number_of_ORL_iterations
            sc_nsc_ORL(k,i)   = scoreClassifier(nsc_labels_ORL{k,i},test_labels_ORL{k});
        end
    end
catch
    disp('Labels not available')
end

%% Nearest Neighbor
try
    sc_nn_MNIST = scoreClassifier(nn_labels_MNIST,test_labels_MNIST(1:testing_subset));
    for k =1:number_of_ORL_iterations
        sc_nn_ORL(k)   = scoreClassifier(nn_labels_ORL{k},test_labels_ORL{k});
    end
catch
    disp('Labels not available')
end
%% BP Perceptron
try
    sc_pbp_MNIST = scoreClassifier(pBP_labels_MNIST,test_labels_MNIST);
    for k =1:number_of_ORL_iterations
        sc_pbp_ORL(k)   = scoreClassifier(pBP_labels_ORL{k},test_labels_ORL{k});
    end
catch
    disp('Labels not available')
end
%% MSE Perceptron
try
    sc_pmse_MNIST = scoreClassifier(pMSE_labels_MNIST,test_labels_MNIST);
    for k =1:number_of_ORL_iterations
        sc_pmse_ORL(k)   = scoreClassifier(pMSE_labels_ORL{k},test_labels_ORL{k});
    end
catch
    disp('Labels not available')
end
%%
if do_PCA == true
    % PCA-reduced Data
    for i = 1:length(target_dimension)
        % Nearest Centroid
        try
            sc_nc_MNIST_pca(i) = scoreClassifier(nc_labels_MNIST_pca{i},test_labels_MNIST);
            for k =1:number_of_ORL_iterations
                sc_nc_ORL_pca(k,i)   = scoreClassifier(nc_labels_ORL_pca{k,i},test_labels_ORL{k});
            end
        catch
            disp('Labels not available')
        end
        try
            % Nearest Subclass
            for k = 1:length(subclasses)
                sc_nsc_MNIST_pca(i,k) = scoreClassifier(nsc_labels_MNIST_pca{k,i},test_labels_MNIST);
                for j =1:number_of_ORL_iterations
                    sc_nsc_ORL_pca(j,i,k)   = scoreClassifier(nsc_labels_ORL_pca{j,k,i},test_labels_ORL{j});
                end
            end
        catch
            disp('Labels not available')
        end
        try
            % Nearest Neighbor
            sc_nn_MNIST_pca(i) = scoreClassifier(nn_labels_MNIST_pca{i},test_labels_MNIST(1:testing_subset));
            for k =1:number_of_ORL_iterations
                sc_nn_ORL_pca(k,i)   = scoreClassifier(nn_labels_ORL_pca{k,i},test_labels_ORL{k});
            end
        catch
            disp('Labels not available')
        end
        try
            % BP Perceptron
            sc_pbp_MNIST_pca(i) = scoreClassifier(pBP_labels_MNIST_pca{i},test_labels_MNIST);
            for k =1:number_of_ORL_iterations
                sc_pbp_ORL_pca(k,i)   = scoreClassifier(pBP_labels_ORL_pca{k,i},test_labels_ORL{k});
            end
        catch
            disp('Labels not available')
        end
        try
            % MSE Perceptron
            sc_pmse_MNIST_pca(i) = scoreClassifier(pMSE_labels_MNIST_pca{i},test_labels_MNIST);
            for k =1:number_of_ORL_iterations
                sc_pmse_ORL_pca(k,i)   = scoreClassifier(pMSE_labels_ORL_pca{k,i},test_labels_ORL{k});
            end
        catch
            disp('Labels not available')
        end
    end
end

disp('Done!')

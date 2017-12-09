%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to apply NCC on data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('Classify using Nearest Neighbor...')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('Original Data...')
% Classification on Raw Image Data

tic;
nn_labels_MNIST = nnClassifier(train_images_MNIST(:,1:training_subset),...
                               test_images_MNIST(:,1:testing_subset),...
                               train_labels_MNIST(1:training_subset));
t_nn_MNIST = toc;

for k = 1:number_of_ORL_iterations
tic;
nn_labels_ORL{k} = nnClassifier(train_images_ORL{k},...
                               test_images_ORL{k},...
                               train_labels_ORL{k});
t_nn_ORL{k} = toc;
end

if do_PCA == true
% Classification on PCA-reduced Image Data
disp('PCA reduced data...')
nn_labels_MNIST_pca = cell(1,length(target_dimension));
nn_labels_ORL_pca = cell(number_of_ORL_iterations,length(target_dimension));

t_nn_MNIST_pca = cell(1,length(target_dimension));
t_nn_ORL_pca = cell(number_of_ORL_iterations,length(target_dimension));

for i= 1:length(target_dimension)
    disp(['Target Dimension: ', num2str(target_dimension(i))])
    tic;
    nn_labels_MNIST_pca{i} = nnClassifier(train_images_MNIST_pca{i}(:,1:training_subset),...
                               test_images_MNIST_pca{i}(:,1:testing_subset),...
                               train_labels_MNIST(1:training_subset));
    t_nn_MNIST_pca{i} = toc;
    
    for k = 1:number_of_ORL_iterations
    tic;
    nn_labels_ORL_pca{k,i} = nnClassifier(train_images_ORL_pca{k,i},...
                               test_images_ORL_pca{k,i},...
                               train_labels_ORL{k});
    t_nn_ORL_pca{k,i} = toc;
    end

end
end
disp('Done!')

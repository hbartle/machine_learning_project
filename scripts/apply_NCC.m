%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to apply NCC on data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('Classify using Nearest Centroid...')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('Original Data...')

tic;
% Classification on Raw Image Data
nc_labels_MNIST = ncClassifier(train_images_MNIST(:,:),...
                               test_images_MNIST(:,:),...
                               train_labels_MNIST(:),...
                               'MNIST');
t_nc_MNIST = toc;

for k = 1:number_of_ORL_iterations
tic;
nc_labels_ORL{k} = ncClassifier(train_images_ORL{k},...
                             test_images_ORL{k},...
                             train_labels_ORL{k},...
                             'ORL');
t_nc_ORL{k} = toc;
end
disp('Done!')

if do_PCA == true
% Classification on PCA-reduced Image Data
disp('PCA reduced data...')
nc_labels_MNIST_pca = cell(1,length(target_dimension));
nc_labels_ORL_pca = cell(number_of_ORL_iterations,length(target_dimension));

t_nc_MNIST_pca = cell(1,length(target_dimension));
t_nc_ORL_pca = cell(number_of_ORL_iterations,length(target_dimension));

for i= 1:length(target_dimension)
    disp(['Target Dimension: ', num2str(target_dimension(i))])
    tic;
    nc_labels_MNIST_pca{i} = ncClassifier(train_images_MNIST_pca{i},...
                                       test_images_MNIST_pca{i},...
                                       train_labels_MNIST,...
                                       'MNIST');
    t_nc_MNIST_pca{i} = toc;
    for k = 1:number_of_ORL_iterations
    tic;
    nc_labels_ORL_pca{k,i} = ncClassifier(train_images_ORL_pca{k,i},...
                                     test_images_ORL_pca{k,i},...
                                     train_labels_ORL{k},...
                                     'ORL');
    t_nc_ORL_pca{k,i} = toc;
    end
end
disp('Done!')
end
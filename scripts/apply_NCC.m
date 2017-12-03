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

tic;
nc_labels_ORL = ncClassifier(train_images_ORL,...
                             test_images_ORL,...
                             train_labels_ORL,...
                             'ORL');
t_nc_ORL = toc;
disp('Done!')


% Classification on PCA-reduced Image Data
disp('PCA reduced data...')
nc_labels_MNIST_pca = cell(1,length(target_dimension));
nc_labels_ORL_pca = cell(1,length(target_dimension));

t_nc_MNIST_pca = nan*ones(1,length(target_dimension));
t_nc_ORL_pca = nan*ones(1,length(target_dimension));

for i= 1:length(target_dimension)
    disp(['Target Dimension: ', num2str(target_dimension(i))])
    tic;
    nc_labels_MNIST_pca{i} = ncClassifier(train_images_MNIST_pca{i},...
                                       test_images_MNIST_pca{i},...
                                       train_labels_MNIST,...
                                       'MNIST');
    t_nc_MNIST_pca(i) = toc;
    tic;
    nc_labels_ORL_pca{i} = ncClassifier(train_images_ORL_pca{i},...
                                     test_images_ORL_pca{i},...
                                     train_labels_ORL,...
                                     'ORL');
    t_nc_ORL_pca(i) = toc;

end

disp('Done!')
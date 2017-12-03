%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to apply PCA on data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Applying PCA...')
target_dimension = [1:10,15:5:40];

train_images_MNIST_pca = cell(1,length(target_dimension));
test_images_MNIST_pca = cell(1,length(target_dimension));
train_images_ORL_pca = cell(1,length(target_dimension));
test_images_ORL_pca = cell(1,length(target_dimension));
 
for i= 1:length(target_dimension)

    disp(['Target Dimension: ', num2str(target_dimension(i))])

    train_images_MNIST_pca{i} = principalComponents(train_images_MNIST,...
                                                   target_dimension(i));
    test_images_MNIST_pca{i} = principalComponents(test_images_MNIST,...
                                                  target_dimension(i));
    train_images_ORL_pca{i} = principalComponents(train_images_ORL,...
                                                 target_dimension(i));
    test_images_ORL_pca{i} = principalComponents(test_images_ORL,...
                                                target_dimension(i));                                     
end
                                     
disp('Done!')
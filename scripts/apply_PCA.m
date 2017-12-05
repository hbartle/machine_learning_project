%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to apply PCA on data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Applying PCA...')

train_images_MNIST_pca = cell(1,length(target_dimension));
test_images_MNIST_pca = cell(1,length(target_dimension));
train_images_ORL_pca = cell(1,length(target_dimension));
test_images_ORL_pca = cell(1,length(target_dimension));
 
for i= 1:length(target_dimension)

    disp(['Target Dimension: ', num2str(target_dimension(i))])

    [w,m] = getPrincipalComponents(train_images_MNIST,target_dimension(i));
    
    train_images_MNIST_pca{i} =transformSamples(train_images_MNIST,...
                                                   w,m);
    test_images_MNIST_pca{i} = transformSamples(test_images_MNIST,...
                                                w,m);
                                             
    [w,m] = getPrincipalComponents(train_images_ORL,target_dimension(i));
    
    train_images_ORL_pca{i} = transformSamples(train_images_ORL,...
                                                   w,m);
    test_images_ORL_pca{i} = transformSamples(test_images_ORL,...
                                                w,m);                                    
end
                                     
disp('Done!')
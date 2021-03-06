%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to apply BP-trained Perceptron on data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('Classify using BP-Perceptron ...')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('Original Data...')


tic;
W_MNIST = trainPerceptronBP(train_images_MNIST,...
                      train_labels_MNIST,...
                      0.1,...
                      'MNIST');           
pBP_labels_MNIST = perceptronBP(W_MNIST,test_images_MNIST,'MNIST');
t_pbp_MNIST = toc;

for k = 1:number_of_ORL_iterations
tic;                  
W_ORL = trainPerceptronBP(train_images_ORL{k},...
                      train_labels_ORL{k},...
                      0.1,...
                      'ORL'); 
pBP_labels_ORL{k} = perceptronBP(W_ORL,test_images_ORL{k},'ORL');
t_pbp_ORL{k} = toc;
end

if do_PCA == true
% Classification on PCA-reduced Image Data
disp('PCA reduced data...')
pBP_labels_MNIST_pca = cell(1,length(target_dimension));
pBP_labels_ORL_pca = cell(number_of_ORL_iterations,length(target_dimension));

t_pbp_MNIST_pca = cell(1,length(target_dimension));
t_pbp_ORL_pca = cell(number_of_ORL_iterations,length(target_dimension));

for i= 1:length(target_dimension)
    disp(['Target Dimension: ', num2str(target_dimension(i))])

    tic;
    W_MNIST_pca = trainPerceptronBP(train_images_MNIST_pca{i},...
                          train_labels_MNIST,...
                          0.1,...
                          'MNIST');              
    pBP_labels_MNIST_pca{i} = perceptronBP(W_MNIST_pca,test_images_MNIST_pca{i},'MNIST');
    t_pbp_MNIST_pca{i} = toc;
    
    for k = 1:number_of_ORL_iterations
    tic;
    W_ORL_pca = trainPerceptronBP(train_images_ORL_pca{k,i},...
                          train_labels_ORL{k},...
                          0.1,...
                          'ORL'); 
    pBP_labels_ORL_pca{k,i} = perceptronBP(W_ORL_pca,test_images_ORL_pca{k,i},'ORL');
    t_pbp_ORL_pca{k,i} = toc;
    end
end
end
disp('Done!')
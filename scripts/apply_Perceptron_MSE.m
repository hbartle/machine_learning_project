%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to apply MSE-trained Perceptron on data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('Classify using MSE-Perceptron ...')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('Original Data...')
tic;
W_MNIST_MSE = trainPerceptronMSE(train_images_MNIST,...
    train_labels_MNIST,'MNIST');
pMSE_labels_MNIST = perceptronMSE(W_MNIST_MSE,test_images_MNIST,'MNIST');
t_pmse_MNIST = toc;

for k = 1:number_of_ORL_iterations
    tic;
    W_ORL_MSE = trainPerceptronMSE(train_images_ORL{k},...
        train_labels_ORL{k},'ORL');
    pMSE_labels_ORL{k} = perceptronMSE(W_ORL_MSE,test_images_ORL{k},'ORL');
    t_pmse_ORL{k} = toc;
end

if do_PCA == true
    % Classification on PCA-reduced Image Data
    disp('PCA reduced data...')
    pMSE_labels_MNIST_pca = cell(1,length(target_dimension));
    pMSE_labels_ORL_pca = cell(number_of_ORL_iterations,length(target_dimension));
    
    t_pmse_MNIST_pca = cell(1,length(target_dimension));
    t_pmse_ORL_pca = cell(number_of_ORL_iterations,length(target_dimension));
    
    for i= 1:length(target_dimension)
        disp(['Target Dimension: ', num2str(target_dimension(i))])
        tic;
        W_MNIST_MSE_pca = trainPerceptronMSE(train_images_MNIST_pca{i},...
            train_labels_MNIST,'MNIST');
        pMSE_labels_MNIST_pca{i} = perceptronMSE(W_MNIST_MSE_pca,...
            test_images_MNIST_pca{i},'MNIST');
        t_pmse_MNIST_pca{i} = toc;
        for k = 1:number_of_ORL_iterations
            tic;
            W_ORL_MSE_pca = trainPerceptronMSE(train_images_ORL_pca{k,i},...
                train_labels_ORL{k},'ORL');
            pMSE_labels_ORL_pca{k,i} = perceptronMSE(W_ORL_MSE_pca,test_images_ORL_pca{k,i},'ORL');
            t_pmse_ORL_pca{k,i} = toc;
        end
    end
end
disp('Done!')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to apply NSC on data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('Classify using Nearest Subclass...')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('Original Data...')
nsc_labels_MNIST = cell(length(subclasses),1);
nsc_labels_ORL = cell(number_of_ORL_iterations, length(subclasses),1);
t_nsc_MNIST = cell(length(subclasses),1);
t_nsc_ORL = cell(number_of_ORL_iterations, length(subclasses),1);

if do_PCA == true
    nsc_labels_MNIST_pca = cell(length(subclasses),length(target_dimension));
    nsc_labels_ORL_pca = cell(number_of_ORL_iterations, length(subclasses),length(target_dimension));
    
    t_nsc_MNIST_pca = cell(length(subclasses),length(target_dimension));
    t_nsc_ORL_pca = cell(number_of_ORL_iterations, length(subclasses),length(target_dimension));
end


for k=1:length(subclasses)
    disp([num2str(subclasses(k)),' Subclasses'])
    
    % Classification on Raw Image Data
    tic;
    nsc_labels_MNIST{k} = nscClassifier(train_images_MNIST,...
        test_images_MNIST,...
        train_labels_MNIST,...
        subclasses(k),...
        'MNIST');
    t_nsc_MNIST{k} = toc;
    for j = 1:number_of_ORL_iterations
        tic;
        nsc_labels_ORL{j,k} = nscClassifier(train_images_ORL{j},...
            test_images_ORL{j},...
            train_labels_ORL{j},...
            subclasses(k),...
            'ORL');
        t_nsc_ORL{j,k} = toc;
    end
    
    if do_PCA == true
        % Classification on PCA-reduced Image Data
        disp('PCA reduced data...')
        
        try
            for i= 1:length(target_dimension)
                disp(['Target Dimension: ', num2str(target_dimension(i))])
                tic;
                nsc_labels_MNIST_pca{k,i} = nscClassifier(train_images_MNIST_pca{i},...
                    test_images_MNIST_pca{i},...
                    train_labels_MNIST,...
                    subclasses(k),...
                    'MNIST');
                t_nsc_MNIST_pca{k,i} = toc;
                for j = 1:number_of_ORL_iterations
                    tic;
                    nsc_labels_ORL_pca{j,k,i} = nscClassifier(train_images_ORL_pca{j,i},...
                        test_images_ORL_pca{j,i},...
                        train_labels_ORL{j},...
                        subclasses(k),...
                        'ORL');
                    t_nsc_ORL_pca{j,k,i} = toc;
                end
                
            end
        catch
            disp('Classification not possible')
        end
    end
end
disp('Done!')

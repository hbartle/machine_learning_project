%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Machine Learning Project Assignment
%   Comparison of Classifiers
% 
%   Optimization and Data Analytics (E17)
%   Aarhus University
%
%   Hannes Bartle
%   hannes.bartle@uni.au.dk
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc

%% Set Path Variable
disp('Setting path...')
addpath('functions/')
addpath('scripts/')
addpath('classifiers/')
addpath(genpath('data/'))
addpath('plots/')
disp('Done!')
%% Load the MNIST Image Data
disp('Loading MNIST data Set...')
train_images_MNIST = loadMNISTImages('train-images.idx3-ubyte');
test_images_MNIST = loadMNISTImages('t10k-images.idx3-ubyte');
train_labels_MNIST = loadMNISTLabels('train-labels.idx1-ubyte');
test_labels_MNIST = loadMNISTLabels('t10k-labels.idx1-ubyte');
disp('Done!')
%% Load the ORL Image Data
disp('Loading ORL data set...')
load('orl_data.mat')
load('orl_lbls.mat')
images_ORL = data;
clear data
labels_ORL = lbls;
clear lbls

% Split up ORL Data
[~,number_of_images]= size(images_ORL);

train_percentage = 0.7;
train_number = 0.7*number_of_images;
test_number = number_of_images - train_number;

% Random Permutation
perm = randperm(number_of_images);

% Shuffle Images
number_of_ORL_iterations = 10;
for k = 1:number_of_ORL_iterations
    
train_images_ORL{k} = [];
test_images_ORL{k} =[];
train_labels_ORL{k} = [];
test_labels_ORL{k} = [];
for i=1:40
    idx = find(labels_ORL ==i);
    perm = idx(randperm(length(idx)));
    train_images_ORL{k} = [train_images_ORL{k} images_ORL(:,perm(1:7))];
    test_images_ORL{k} = [test_images_ORL{k} images_ORL(:,perm(8:10))];
    train_labels_ORL{k} = [train_labels_ORL{k} labels_ORL(perm(1:7))'];
    test_labels_ORL{k} = [test_labels_ORL{k} labels_ORL(perm(8:10))'];
end
train_labels_ORL{k} = train_labels_ORL{k}';
test_labels_ORL{k} = test_labels_ORL{k}';
end
disp('Done!')

%% Dimensionality Reduction using PCA
do_PCA = true;
target_dimension = 1:50;
if do_PCA == true
    apply_PCA
end
%% Nearest Centroid Classifier Evaluation
apply_NCC
%% Nearest Subclass Classifier
subclasses = [2,3,5];
apply_NSC
%% Nearest Neighborhood Classifier
training_subset = 60000;
testing_subset = 10000;
apply_NN
%% Perceptron with Backpropagation on original Data
apply_Perceptron_BP
%% Perceptron Minimum Square Error
apply_Perceptron_MSE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Scoring
score
%% Plots
close all
FontSize= 20;
LineWidth = 2;


fig_MNIST = figure('units','normalized','outerposition',[0 0 1 1]);
y_MNIST = [sc_nc_MNIST;...
     sc_nsc_MNIST(1);...
     sc_nsc_MNIST(2);...
     sc_nsc_MNIST(3);...
     sc_nn_MNIST;...
     sc_pbp_MNIST;...
     sc_pmse_MNIST]*100;
c = {'NC','NSC2','NSC3','NSC5','NN','P-BP','P-MSE'};
bar(y_MNIST)
barvalues;
ylim([0 120]);
set(gca,'xticklabel',c)
set(gca,'FontSize',FontSize);
ylabel('Success Rate [%]','FontSize',FontSize)
grid on


fig_ORL = figure('units','normalized','outerposition',[0 0 1 1]);
y_MNIST = [sc_nc_ORL;...
     sc_nsc_ORL(1);...
     sc_nsc_ORL(2);...
     sc_nsc_ORL(3);...
     sc_nn_ORL;...
     sc_pbp_ORL;...
     sc_pmse_ORL]*100;
c = {'NC','NSC2','NSC3','NSC5','NN','P-BP','P-MSE'};
bar(y_MNIST)
barvalues;
ylim([0 120]);
set(gca,'xticklabel',c)
set(gca,'FontSize',FontSize);
ylabel('Success Rate [%]','FontSize',FontSize)
grid on

fig_MNIST_time = figure('units','normalized','outerposition',[0 0 1 1]);
y_MNIST = [t_nc_MNIST;...
     t_nsc_MNIST{1};...
     t_nsc_MNIST{2};...
     t_nsc_MNIST{3};...
     t_nn_MNIST;...
     t_pbp_MNIST;...
     t_pmse_MNIST]*1000;
c = {'NC','NSC2','NSC3','NSC5','NN','P-BP','P-MSE'};
bar(y_MNIST)
barvalues;
set(gca,'xticklabel',c)
set(gca,'FontSize',FontSize);
ylabel('Execution Time [ms]','FontSize',FontSize)
grid on

fig_ORL_time = figure('units','normalized','outerposition',[0 0 1 1]);
y_MNIST = [t_nc_ORL;...
     t_nsc_ORL{1};...
     t_nsc_ORL{2};...
     t_nsc_ORL{3};...
     t_nn_ORL;...
     t_pbp_ORL;...
     t_pmse_ORL]*1000;
c = {'NC','NSC2','NSC3','NSC5','NN','P-BP','P-MSE'};
bar(y_MNIST)
barvalues;
set(gca,'xticklabel',c)
set(gca,'FontSize',FontSize);
ylabel('Execution Time [ms]','FontSize',FontSize)
grid on

if do_PCA == true
fig_pca_MNIST = figure('units','normalized','outerposition',[0 0 1 1]);
h_pca_mnist = plot(target_dimension,sc_nc_MNIST_pca*100,...
                     target_dimension,sc_nsc_MNIST_pca(:,1)*100,...
                     target_dimension,sc_nsc_MNIST_pca(:,2)*100,...
                     target_dimension,sc_nsc_MNIST_pca(:,3)*100,...
                     target_dimension,sc_nn_MNIST_pca*100,...
                     target_dimension,sc_pbp_MNIST_pca*100,...
                     target_dimension,sc_pmse_MNIST_pca*100);
            
set(h_pca_mnist(1),'Linewidth',LineWidth);
set(h_pca_mnist(2),'Linewidth',LineWidth);
set(h_pca_mnist(3),'Linewidth',LineWidth);
set(h_pca_mnist(4),'Linewidth',LineWidth);
set(h_pca_mnist(5),'Linewidth',LineWidth);
set(h_pca_mnist(6),'Linewidth',LineWidth);
set(h_pca_mnist(7),'Linewidth',LineWidth);

legend('NC','NSC2','NSC3','NSC5','NN','P-BP','P-MSE','Location','SE');
xlabel('Dimensions')
set(gca,'FontSize',FontSize);
ylabel('Success Rate [%]','FontSize',FontSize)
grid on


fig_pca_ORL = figure('units','normalized','outerposition',[0 0 1 1]);
h_pca_orl = plot(target_dimension,sc_nc_ORL_pca*100,...
                 target_dimension,sc_nsc_ORL_pca(:,1)*100,...
                 target_dimension,sc_nsc_ORL_pca(:,2)*100,...
                 target_dimension,sc_nsc_ORL_pca(:,3)*100,...
                 target_dimension,sc_nn_ORL_pca*100,...
                 target_dimension,sc_pbp_ORL_pca*100,...
                 target_dimension,sc_pmse_ORL_pca*100);
 
set(h_pca_orl(1),'Linewidth',LineWidth);
set(h_pca_orl(2),'Linewidth',LineWidth);
set(h_pca_orl(3),'Linewidth',LineWidth);
set(h_pca_orl(4),'Linewidth',LineWidth);
set(h_pca_orl(5),'Linewidth',LineWidth);
set(h_pca_orl(6),'Linewidth',LineWidth);
set(h_pca_orl(7),'Linewidth',LineWidth); 
 
legend('NC','NSC2','NSC3','NSC5','NN','P-BP','P-MSE','Location','SE');
xlabel('Dimensions')
set(gca,'FontSize',FontSize);
ylabel('Success Rate [%]','FontSize',FontSize)
grid on

end

fig_performance = figure('units','normalized','outerposition',[0 0 1 1]);
x_MNIST = [sc_nc_MNIST,sc_nsc_MNIST, sc_nn_MNIST, sc_pbp_MNIST,sc_pmse_MNIST]'*100;
y_MNIST = [t_nc_MNIST,cell2mat(t_nsc_MNIST)',t_nn_MNIST,t_pbp_MNIST,t_pmse_MNIST]';
x_ORL = [sc_nc_ORL,sc_nsc_ORL, sc_nn_ORL, sc_pbp_ORL,sc_pmse_ORL]'*100;
y_ORL = [t_nc_ORL,cell2mat(t_nsc_ORL)',t_nn_ORL,t_pbp_ORL,t_pmse_ORL]';

h=semilogy(x_MNIST,y_MNIST,'x',...
         x_ORL,y_ORL,'o');
lgd = legend('MNIST','ORL');
lgd.FontSize = FontSize;
l = {'NC','NSC2','NSC3','NSC5','NN','P-BP','P-MSE'};
labelpoints(x_MNIST,y_MNIST,l,'N','FontSize',FontSize);
labelpoints(x_ORL,y_ORL,l,'N','FontSize',FontSize);
set(h(1),'MarkerSize',12,'Linewidth',3);
set(h(2),'MarkerSize',12,'Linewidth',3);
set(gca,'FontSize',FontSize);
ylabel('Execution Time [ms]','FontSize',FontSize)
xlabel('Success Rate [%]','Fontsize',FontSize)
grid on

fig_performance_pca = figure('units','normalized','outerposition',[0 0 1 1]);
x_MNIST_pca = [sc_nc_MNIST_pca(20),sc_nsc_MNIST_pca(20,:), sc_nn_MNIST_pca(20), sc_pbp_MNIST_pca(20),sc_pmse_MNIST_pca(20)]'*100;
y_MNIST_pca = [t_nc_MNIST_pca(20),t_nsc_MNIST_pca{1,20},t_nsc_MNIST_pca{2,20},t_nsc_MNIST_pca{3,20},t_nn_MNIST_pca(20),t_pbp_MNIST_pca(20),t_pmse_MNIST_pca(20)]';
x_ORL_pca = [sc_nc_ORL_pca(20),sc_nsc_ORL_pca(20,:), sc_nn_ORL_pca(20), sc_pbp_ORL_pca(20),sc_pmse_ORL_pca(20)]'*100;
y_ORL_pca = [t_nc_ORL_pca(20),t_nsc_ORL_pca{1,20},t_nsc_ORL_pca{2,20},t_nsc_ORL_pca{3,20},t_nn_ORL_pca(20),t_pbp_ORL_pca(20),t_pmse_ORL_pca(20)]';
h=semilogy(x_MNIST_pca,y_MNIST_pca,'s',...
         x_ORL_pca,y_ORL_pca,'d');
lgd = legend('MNIST','ORL','Location','NW');
lgd.FontSize = FontSize;
l = {'NC','NSC2','NSC3','NSC5','NN','P-BP','P-MSE'};
labelpoints(x_MNIST_pca,y_MNIST_pca,l,'N','FontSize',FontSize);
labelpoints(x_ORL_pca,y_ORL_pca,l,'N','FontSize',FontSize);
set(h(1),'MarkerSize',12,'Linewidth',3);
set(h(2),'MarkerSize',12,'Linewidth',3);
set(gca,'FontSize',FontSize);
ylabel('Execution Time [ms]','FontSize',FontSize)
xlabel('Success Rate [%]','Fontsize',FontSize)
grid on

%% Save Figures
% print(fig_MNIST, 'plots/mnist_success','-depsc');
% print(fig_MNIST_time, 'plots/mnist_time','-depsc');
% print(fig_ORL, 'plots/orl_success','-depsc');
% print(fig_ORL_time, 'plots/orl_time','-depsc');
% print(fig_pca_MNIST, 'plots/mnist_pca','-depsc');
% print(fig_pca_ORL, 'plots/orl_pca','-depsc');
%print(fig_performance, 'plots/comparison_performance','-depsc');
%print(fig_performance_pca, 'plots/comparison_performance_pca','-depsc');

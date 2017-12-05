function [ transformed_samples ] = transformSamples( samples,W,m )
%TRANSFORMSAMPLES Summary of this function goes here
%   Detailed explanation goes here

[~,number_of_samples] = size(samples);

X = (samples-m*ones(1,number_of_samples))';
transformed_samples = (X*W)';


end


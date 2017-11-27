function [success_rate,correct_samples ] = scoreClassifier( labels_classified, test_labels )
%SCORECLASSIFIER Score a Classifier

total_samples = length(test_labels);

correct_samples = find(labels_classified == test_labels');

success_rate = length(correct_samples)/total_samples;

end


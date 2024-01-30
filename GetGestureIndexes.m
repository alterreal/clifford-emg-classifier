function [gesture_indexes] = GetGestureIndexes(n_examples, gesture_classes, dataset)
%GetGestureIndexes is a function that fetches the indexes of n random 
%examples of each input class from the provided dataset.
%
%INPUT
%       - n_examples: desired number of samples for each class
%       - gesture_classes: vector containing the target class indexes
%       - dataset: dataset organized as on samples per row containing the
%       features followed by the respective class
%
%OUTPUT
%       - gesture_indexes: indexes of the total random samples
%
%--------------------------------------------------------------------------
% Alexandre Calado
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    gesture_indexes = zeros(length(gesture_classes),n_examples);
    for i = 1:length(gesture_classes)
        class_indices = find(dataset(:,end)==gesture_classes(i));
        idx = randperm(size(class_indices,1),n_examples) ;
        gesture_indexes(i,:) = class_indices(idx) ;
    end
    gesture_indexes = reshape(gesture_indexes,(length(gesture_classes))*n_examples,1);
end


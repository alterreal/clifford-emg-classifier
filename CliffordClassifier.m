function [uPClassifier] = CliffordClassifier(X_train, Y_train, calculate_abs_th, centroid, X_val, Y_val, abstention_threshold_range)
%
%CliffordClassifier is a function that constructs a Clifford algebra based
%algorithm (uP) for EMG gesture classification.
%
% INPUTS:
%       - X_train: training set data, where each row is a sample composed
%       of n features (n electrode channels)
%       - Y_train: training classes, including rest (0)
%       - n: number of polygon sides (i.e. number of sensors). Only 6 or 8.
%       - calculate_abs_th : bool to decide if abstentions thresholds are
%       computed (true: yes; false: no)
%       - X_val (OPTIONAL): validation set data, where each row is a sample 
%       composed of n features (n electrode channels). Used to compute 
%       abstentions thresholds.
%       - Y_val (OPTIONAL): validation classes, excluding rest (0)
%       - abstention_threshold_range (OPTIONAL): vector containing the 
%       possible abstention thresholds.
%
% OUTPUTS:
%       - uPClassifier: struct containing the training data, training 
%       classes, rest thresholds, training set first shapes, maximum value
%       in the training set (for scaling), rest thresholds and abstentions
%       thresholds (if calculate_abs_th is true)
%--------------------------------------------------------------------------
% Alexandre Calado
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if calculate_abs_th && nargin < 6
        error('Missing arguments for abstention threshold calculations')
    end
    
    %Save number of sensors (features) ( current implementation supports 
    %6 or 8)
    n = size(X_train,2); 
    
    if n~=6 && n~=8
        error('CliffordClassifier only supports data from 6 or 8 sensors (6 or 8 features)')
    end

    %Extract maximum channel value from all training examples for scaling
    training_set_max = max(max((X_train))); 
    X_train = X_train./training_set_max; %scale training set data
    
    %Find the indexes of all classes = 0 (rest) and create a "rest set"
    ixR = Y_train==0;
    %ixR = find(Y_train==0);
    X_rest = X_train(ixR,:);
    %Dispose of rest gestures in training set
    %X_train(ixR,:) = [];
    %Y_train(ixR,:) = [];
    X_train = X_train(~ixR,:);
    Y_train = Y_train(~ixR);
    
    %Compute the rest threshold
    rest_th = max(X_rest);
    
    %Calculate first shapes for each training example
    [X_train_XS, X_train_YS] = CalculateFirstShapes(X_train, n);
    
    classes = unique(Y_train)';
    abstention_thresholds = zeros(1,length(classes)); %abstention threshold per class
    
    
    
    %In case that abstention thresholds must be calculated
    if calculate_abs_th
        X_val = X_val./training_set_max; %scale validation set data
        %Calculate abstention threshold separately for each class (except "rest")
        for i_c = 1:length(classes)
            accuracy_val_max = -1;
            for i_a = 1:length(abstention_threshold_range)
                %Evaluate only on one class from the validation data to
                %compute respective abstention threshold
                class_indexes = Y_val == i_c;
                X_val_single_class = X_val(class_indexes,:);
                Y_val_single_class = Y_val(class_indexes);
                Y_hyp_val_single_class = zeros(size(Y_val_single_class));
                for i_val = 1:size(X_val_single_class,1)
                    
                    GD = CalculateGDuP(X_val_single_class(i_val,:), X_train,X_train_XS,X_train_YS, n);
                    [GD_min, GD_min_idx] = min(GD);
                    if GD_min < abstention_threshold_range(i_a)
                        Y_hyp_val_single_class(i_val) = Y_train(GD_min_idx);
                    elseif GD_min >= abstention_threshold_range(i_a)
                        Y_hyp_val_single_class(i_val) = 0;
                    end
                end
                abstention_val = length(Y_hyp_val_single_class(Y_hyp_val_single_class==5))/length(Y_hyp_val_single_class); 
                %Discard abstentions before calculating accuracy
                Y_val_single_class(Y_hyp_val_single_class==0) = [];
                Y_hyp_val_single_class(Y_hyp_val_single_class==0) = [];
                accuracy_val = sum(Y_val_single_class == Y_hyp_val_single_class)/length(Y_hyp_val_single_class);            
                if accuracy_val > accuracy_val_max 
                    accuracy_val_max = accuracy_val;
                    abstention_thresholds(i_c) = abstention_threshold_range(i_a);
                end
            end
            %disp(['Abstention threshold of class ', num2str(i_c),': ', num2str(abstention_thresholds(i_c))])
        end
        uPClassifier = struct('TrainingData',X_train,'TrainingClasses',Y_train, ...
        'TrainingFirstShapeXS', X_train_XS, 'TrainingFirstShapeYS', X_train_YS,...
        'RestThreshold', rest_th, 'TrainingSetMax', training_set_max, ...
        'SensorsNumber', n, 'AbstentionThresholds', abstention_thresholds);
    elseif centroid
        X_train_ = [];
        X_train_XS_ = [];
        X_train_YS_ = [];
        for i = 1:length(classes)
            GD = CalculateGDuP(mean(X_train(Y_train == classes(i),:)),... 
                X_train(Y_train == classes(i),:),...
                X_train_XS(Y_train == classes(i),:),...
                X_train_YS(Y_train == classes(i),:), n);
            abstention_thresholds(i) = mean(GD);
            X_train_ = [X_train_; mean(X_train(Y_train == classes(i),:))];
            X_train_XS_ = [X_train_XS_; mean(X_train_XS(Y_train == classes(i),:))];
            X_train_YS_ =  [X_train_YS_;mean(X_train_YS(Y_train == classes(i),:))];
        end
        Y_train = classes;
        X_train = X_train_;
        X_train_XS = X_train_XS_;
        X_train_YS = X_train_YS_;
        
        uPClassifier = struct('TrainingData',X_train,'TrainingClasses',Y_train, ...
        'TrainingFirstShapeXS', X_train_XS, 'TrainingFirstShapeYS', X_train_YS,...
        'RestThreshold', rest_th, 'TrainingSetMax', training_set_max, 'SensorsNumber', n,...
        'AbstentionThresholds', abstention_thresholds);
    else
        abstention_thresholds = zeros(1,4); %abstention threshold per class
        uPClassifier = struct('TrainingData',X_train,'TrainingClasses',Y_train, ...
        'TrainingFirstShapeXS', X_train_XS, 'TrainingFirstShapeYS', X_train_YS,...
        'RestThreshold', rest_th, 'TrainingSetMax', training_set_max, 'SensorsNumber', n,...
        'AbstentionThresholds', abstention_thresholds);
    end 
end


function [y_hyp] = CliffordPredict(x,cliffordClassifier, abstention_strategy, y_hyp_last)
%
%CliffordPredict is a function that predicts the label of the input gesture, 
%based on the provided uP algorithm.
%
% INPUTS:
%       - x: input candidate gesture: |x1|x2|...|xn|, where n is the number
%       of used sensors
%       - cliffordClassifier: structure of the uP algorithm that contains
%       the training set, first shapes and other parameters needed for
%       classification
%       - abstention_strategy: choose the abstention strategy (1 to 4)
%       - y_hyp_last (OPTIONAL): last predicted class, necessary if using
%       abstention strategy 2.
%
% OUTPUTS:
%       - y_hyp: predicted label for x
%--------------------------------------------------------------------------
% Alexandre Calado
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %Get necessary parameters and data
    n = cliffordClassifier.SensorsNumber;
    training_set_max = cliffordClassifier.TrainingSetMax;
    X_train = cliffordClassifier.TrainingData;
    Y_train = cliffordClassifier.TrainingClasses;
    X_train_XS = cliffordClassifier.TrainingFirstShapeXS;
    X_train_YS = cliffordClassifier.TrainingFirstShapeYS;
    rest_th = cliffordClassifier.RestThreshold;
    
    if abstention_strategy == 1 || abstention_strategy == 4 || abstention_strategy == 5
        abstention_thresholds = cliffordClassifier.AbstentionThresholds;
    end
    
    x = x./training_set_max; %scale values
    
    %if the sum of the differences between the value from each 
    %electrode channel and respective rest threshold is lower or
    %equal to zero then abstain
    if sum(x-rest_th) <= 0 
        y_hyp = 0;
    else
        %Find the training example that minimizes GD
        %Compute all GDs (Global Distances), one for each training example
         GD = CalculateGDuP(x, X_train,X_train_XS,X_train_YS, n);

        %Get first minimum GD
        [GD_sorted, GD_idx_sorted] = sort(GD);
        %[GD_min_1, GD_min_idx_1] = min(GD);
        GD_min_1 = GD_sorted(1);
        GD_min_idx_1 = GD_idx_sorted(1);
        
        if abstention_strategy ~= 1
            %Get second minimum GD
            GD_min_idx_2 = GD_idx_sorted(2);
        end
        
        switch abstention_strategy
            case 1
                %If the predicted class GD is lower than than the
                %respective class abstention threshold, do nothing. If the
                %GD is higher than the abstention threshold, then abstain.
                y_hyp = Y_train(GD_min_idx_1);
                if GD_min_1 >= abstention_thresholds(y_hyp)
                    y_hyp = 0;
                end
            case 2
                %If the classes corresponding to the two lowest GDs are the
                %same, then we can be confident to classify the candidate 
                %gesture as that class
                if Y_train(GD_min_idx_1) == Y_train(GD_min_idx_2)
                    y_hyp = Y_train(GD_min_idx_1);
                %Otherwise, label candidate gesture as previously classified class
                else
                    y_hyp = y_hyp_last;
                end
            case 3
                %If the classes corresponding to the two lowest GDs are the
                %same, then we can be confident to classify the candidate 
                %gesture as that class
                if Y_train(GD_min_idx_1) == Y_train(GD_min_idx_2)
                    y_hyp = Y_train(GD_min_idx_1);
                %Otherwise, abstain
                else
                    y_hyp = 0;
                end
            case 4
                y_hyp = Y_train(GD_min_idx_1);
                %If the classes corresponding to the two lowest GDs are the
                %same OR if the predicted class GD is lower than than the
                %respective class abstention threshold, then we can be 
                %confident to classify the candidate gesture as that class
                if Y_train(GD_min_idx_1) == Y_train(GD_min_idx_2) && GD_min_1 <= abstention_thresholds(y_hyp)
                    y_hyp = Y_train(GD_min_idx_1);
                %Otherwise abstain
                else
                    y_hyp =  0;
                end 
            case 5
                y_hyp = Y_train(GD_min_idx_1);
                %If the classes corresponding to the two lowest GDs are the
                %same OR if the predicted class GD is lower than than the
                %respective class abstention threshold, then we can be 
                %confident to classify the candidate gesture as that class
                if GD_min_1 >= abstention_thresholds(y_hyp) || sum(GD_min_1 >= abstention_thresholds) > 1 
                    y_hyp =  0;
                end 
        end    
    end
end


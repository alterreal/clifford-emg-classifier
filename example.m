%This is an example of how to use the uP6 classifier, all parameters here 
%considered are based on the paper "A Geometric Algebra-Based Approach for 
%Myoelectric Pattern Recognition Control and Faster Prosthesis Recalibration"

% Clean up and setup
clear
clc
close all

%Gesture names and respective classes (in current implementantion "rest" must be 0)
gestures = ["Rest","Opening","Closing","Wrist Flexion","Wrist Extension"];
classes = [0 1 2 3 4];
n_sensors = 6; %number of sensors, in current implementation 6 or 8
n_training_examples = 16; %Define number of training examples per class
n_validation_examples = 25; %Define number of validation examples per class
abstention_threshold_range = 0.3:-0.001:0.1; %list of possible abstention thresholds

%Select classifier
clf_name = "uP6_1";

%Define sampling and downsampling frequencies
fs = 1000;
fs_ds = 300;
s=int32(fs/fs_ds);

%Get data
dataset = load("example_data.txt");

%Reorganize order of electrodes to match a hexagon
dataset = dataset(:,[3 1 6 4 2 5 7]); 

%Downsample data to selected downsampling frequency and use that
%data as test set. Store the remaining data in dataset to use for
%training and validation sets
[m,~]=size(dataset);
test_set = dataset(1:s:m,:);
dataset(1:s:m,:) = [];

%Set different seed for random number generator to ensure
%"randomness" of training and validation examples
rng(42)

%Get the indexes of n random examples of each gesture (except
%"rest") to form the training set
training_indices = GetGestureIndexes(n_training_examples, classes, dataset);     
training_set = dataset(training_indices,:);
dataset(training_indices,:) = []; %remove the training set gestures from the data set

%Get the indexes of n random examples of each gesture to form the validation set
validation_indices = GetGestureIndexes(n_validation_examples, classes(2:end), dataset); 
validation_set = dataset(validation_indices,:);
dataset(validation_indices,:) = []; %remove the validation set gestures from the data set

%Separate data and classes
Y_train = training_set(:,end);
X_train = training_set(:,1:end-1);
Y_test = test_set(:,end);
X_test = test_set(:,1:end-1);
Y_val = validation_set(:,end);
X_val = validation_set(:,1:end-1);

%Build classifier, uP6_1 in this case
switch clf_name
    case "uP6_1"
        uP6_1 = CliffordClassifier(X_train, Y_train, true, false, X_val, Y_val, abstention_threshold_range);
    case "uP6_2"
        uP6_2 = CliffordClassifier(X_train, Y_train, false, false);
    case "uP6_3"
        uP6_3 = CliffordClassifier(X_train, Y_train, false, false);
    case "uP6_4"
        uP6_4 = CliffordClassifier(X_train, Y_train, true, false, X_val, Y_val, abstention_threshold_range);
end

%Classify the test set data
Y_hyp = zeros(size(Y_test));
y_hyp_last = 0;
                
for i_test = 1:size(X_test,1)
    switch clf_name
        case "uP6_1"
            Y_hyp(i_test) = CliffordPredict(X_test(i_test,:),uP6_1, 1);
        case "uP6_2"
            Y_hyp(i_test) = CliffordPredict(X_test(i_test,:),uP6_2, 2, y_hyp_last);
            y_hyp_last = Y_hyp(i_test);
        case "uP6_3"
            Y_hyp(i_test) = CliffordPredict(X_test(i_test,:),uP6_3, 3, y_hyp_last);
            y_hyp_last = Y_hyp(i_test);
        case "uP6_4"
            Y_hyp(i_test) = CliffordPredict(X_test(i_test,:),uP6_4, 4, y_hyp_last);
            y_hyp_last = Y_hyp(i_test);                      
    end 
end

%Calculate and save evaluation metrics
[cm, accuracy, abstention, precision,sensitivity,f1_score] = CalculateMetrics(Y_test, Y_hyp);

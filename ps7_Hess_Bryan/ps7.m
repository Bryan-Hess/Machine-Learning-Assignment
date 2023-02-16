%% ps7_Hess_Bryan
clear all;

%% 0.  Read Data 

load 'input/HW7_Data.mat';
load 'input/HW7_weights_2.mat';

%% 1. Forward Propagation

%%ps7-1-a.)
P = predict(Theta1, Theta2, X);

%%ps7-1-b.)
accuracyofP = (mean(P == y)*100);
fprintf('\nAccuracy of P: %.2f%%\n', accuracyofP);

%% 2. Cost Function

%%ps7-2-b.)
J = zeros(3,1);

for i = [0 1 2]
    J(i+1) = nnCost(Theta1, Theta2, X, y, 3, i);
end

J_Table = table([0;1;2], J, 'VariableNames', {'λ', 'J'})


%% 3. Derivation of the Active Function : Sigmoid Gradient

z = [-10 0 10]';
g_prime = sigmoidGradient(z)


%% For part 4 see sGD.m file

%% 5. Testing the Network
%Data to training and testing
indicies = randperm(length(X));
X_train = X(indicies(1:round(length(indicies)*0.85)), :);
y_train = y(indicies(1:round(length(indicies)*0.85)), :);

X_test = X(indicies((round(length(indicies)*0.85))+1:end),:);
y_test = y(indicies((round(length(indicies)*0.85))+1:end));

%sGD variable declaration (as per top of instructions)
num_labels = 3;
input_layer_size = 4;
hidden_layer_size = 8;
alpha = 0.01;

%Initializing trainingAcc and J
J_train = zeros(hidden_layer_size,1);
J_test = zeros(hidden_layer_size,1);
trainingAcc = zeros(hidden_layer_size,1);
testingAcc = zeros(hidden_layer_size,1);

fprintf('\nAlpha: %.3f\n', alpha);

i = 1;
for MaxEpochs = [50 100]
    for lambda = [0 0.01 0.1 1]
        %Thetas
        [Theta1, Theta2] = sGD(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda, alpha, MaxEpochs); 
        
        %Training data pred and acc
        P = predict(Theta1, Theta2, X_train);
        trainingAcc(i) = mean(P==y_train)*100; 
        %Training data cost
        J_train(i) = nnCost(Theta1, Theta2, X_train, y_train, num_labels, lambda);
        
        %Test data pred and acc
        P = predict(Theta1, Theta2, X_test);
        testingAcc(i) = mean(P==y_test)*100;
        %Test data cost
        J_test(i) = nnCost(Theta1, Theta2, X_test, y_test, num_labels, lambda);
        
        i = i+1;    
    end
end

%Setting Up table variables
%%50 Epochs
TrAc50Ep = trainingAcc(1:(hidden_layer_size/2));
TrCo50Ep = J_train(1:(hidden_layer_size/2));
TeAc50Ep = testingAcc(1:(hidden_layer_size/2));
TeCo50Ep = J_test(1:(hidden_layer_size/2));
%%100 Epochs
TrAc100Ep = trainingAcc((hidden_layer_size/2)+1:end);
TrCo100Ep = J_train((hidden_layer_size/2)+1:end);
TeAc100Ep = testingAcc((hidden_layer_size/2)+1:end);
TeCo100Ep = J_test((hidden_layer_size/2)+1:end);

%Printing Tables
Accuracy_table = table([0;0.01;0.1;1], TrAc50Ep, TeAc50Ep, TrAc100Ep, TeAc100Ep,...
    'VariableNames', {'λ','Training Data Accuracy 50 Epochs', 'Testing Data Accuracy 50 Epochs'...
    'Training Data Accuracy 100 Epochs', 'Testing Data Accuracy 100 Epochs'})

Cost_table = table([0;0.01;0.1;1], TrCo50Ep, TeCo50Ep, TrCo100Ep, TeCo100Ep,...
    'VariableNames', {'λ','Training Data Cost 50 Epochs', 'Testing Cost Accuracy 50 Epochs'...
    'Training Data Cost 100 Epochs', 'Testing Data Cost 100 Epochs'})





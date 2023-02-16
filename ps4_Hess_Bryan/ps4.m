%%ps4_Hess_Bryan
clear ; 
close all; 
clc;

%%%%%%%%%%%%%%%%%%%%ps4-1-b.)
fprintf('\n\nps4-1-b.)');
load('input/hw4_data1.mat');
[m, n] = size(X_data); %number of training examples m
X = [ones(length(X_data), 1) X_data];
fprintf('\nSize of X: ');
size(X)
fprintf('\nSize of y: ');
size(y)

%%%%%%%%%%%%%%%%%%%%ps4-1-c.)
lambda = [0, 0.001, 0.003, 0.005, 0.007, 0.009, 0.012, 0.017];

for i = 1:20
    %Splits data into test data and training data
    indicies = randperm(length(X)); 
    X_train = X(indicies(1:round(length(indicies)*0.88)),:);
    y_train = y(indicies(1:round(length(indicies)*0.88)),:);

    X_test = X(indicies((round(length(indicies)*0.88))+1:end),:);
    y_test = y(indicies((round(length(indicies)*0.88))+1:end));
    
    for j = 1:length(lambda)
        theta = Reg_normalEqn(X_train, y_train, lambda(j));
        trainError(i,j) = computeCost(X_train, y_train, theta);
        testError(i,j) = computeCost(X_test, y_test, theta);
    end
end

avgTrainError = mean(trainError);
avgTestError = mean(testError);

%%Plots data
figure(1)
hold on
plot(lambda, avgTrainError, 'r-*')
plot(lambda, avgTestError, 'b-o')
xlabel('𝜆')
ylabel('Average Error')
legend('Training Error', 'Testing Error')
hold off

%%The lamda (1) value should be used as the distance between the testing and
%%the training error is minimized. For this data set that value would be
%%0.001. After this, the two errors diverge. While training error does go
%%down, this only means we are over-relying on the specific testing data. 

%%%%%%%%%%%%%%%%%%%%ps4-2-a.)
fprintf('\n\nps4-2-a.)');
load('input/hw4_data2.mat');

%First Classifier Data
X_train1 = [X1;X2;X3;X4];
y_train1 = [y1;y2;y3;y4];
X_test1 = X5;
y_test1 = y5;

%Second Classifier Data
X_train2 = [X1;X2;X3;X5];
y_train2 = [y1;y2;y3;y5];
X_test2 = X4;
y_test2 = y4;

%Third Classifier Data
X_train3 = [X1;X2;X4;X5];
y_train3 = [y1;y2;y4;y5];
X_test3 = X3;
y_test3 = y3;

%Fourth Classifier Data
X_train4 = [X1;X3;X4;X5];
y_train4 = [y1;y3;y4;y5];
X_test4 = X2;
y_test4 = y2;

%Fifth Classifier Data
X_train5 = [X2;X3;X4;X5];
y_train5 = [y2;y3;y4;y5];
X_test5 = X1;
y_test5 = y1;


avgAcc = [];
i=1;
for k = 1:2:15
    knn = fitcknn(X_train1, y_train1, 'NumNeighbors', k, 'Standardize', 1);
    corr(1,i) = nnz(predict(knn, X_test1) == y_test1);
    
    knn = fitcknn(X_train2, y_train2, 'NumNeighbors', k, 'Standardize', 1);  
    corr(2,i) = nnz(predict(knn, X_test2) == y_test2);
    
    knn = fitcknn(X_train3, y_train3, 'NumNeighbors', k, 'Standardize', 1);    
    corr(3,i) = nnz(predict(knn, X_test3) == y_test3);
    
    knn = fitcknn(X_train4, y_train4, 'NumNeighbors', k, 'Standardize', 1);
    corr(4,i) = nnz(predict(knn, X_test4) == y_test4);
    
    knn = fitcknn(X_train5, y_train5, 'NumNeighbors', k, 'Standardize', 1);   
    corr(5,i) = nnz(predict(knn, X_test5) == y_test5);
    
    i=i+1;
end

%%Use length of  y_test1 becasuse they are all the same
avgAcc=mean(corr)/length(y_test1);

hold on

figure(2)
plot((1:2:15), avgAcc);

xlabel('K')
ylabel('Accuracy')

hold off

%%For this dataset I reccomend using a K of 9, given it has the highest
%%yeilded accuracy. This value is not robust to other problems as KNN is
%%very data specific. Some datasets might need a larger or smaller K
%%depending on how the data is grouped.

%%%%%%%%%%%%%%%%%%%%ps4-3-a.)
%%MADE weightedKNN FUCNTION

%%%%%%%%%%%%%%%%%%%%ps4-3-b.)
load('input/hw4_data3.mat');
sigma = [0.01 0.1 0.5 1 3 5];
acc = [0 0 0 0 0 0];

sigma = [0.01, 0.1, 0.5, 1, 3, 5];
acc = [0, 0, 0, 0, 0, 0];

k = 0;
for i = sigma
    [y_predict] = weightedKNN(X_train, y_train, X_test, i);  
    for j = 1:25
        if y_predict(j) == y_test(j)
            acc(k+1) = (acc(k+1) + 1);
        end
    end
    k = k+1;  
end
acc = acc./size(y_test,1)

VarNames = {'Sigma', 'Accuracy'};
table(sigma', acc', 'VariableNames',VarNames)

%%The scaling in the distance metric affects the regional shapes. Too
%%small a scaling value and the data is too broad, but too high and it's
%%too specific. A signma that adjusts the regions correctly should be
%%chosen


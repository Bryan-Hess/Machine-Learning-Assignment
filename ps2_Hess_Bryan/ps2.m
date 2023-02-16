%%ps2_Hess_Bryan

%%ps2-1.)
fprintf('ps2-1.');
%initialize test data
x0 = [1,1,1,1]';
x1 = [1,2,3,4]';
x2 = [1,2,3,4]';
y = [2,4,6,8]';
theta1 = [0, 1, 0.5]';
theta2 = [3.5, 0, 0]';
X=[x0, x1, x2];

J1 = computeCost(X, y, theta1);
J2 = computeCost(X, y, theta2);

fprintf('\nCost 1: %d\nCost 2: %d', J1, J2);

%%ps2-2.)
fprintf('\n\nps2-2.');
iters = 15;
alpha = 0.01;
     
[theta, cost] = gradientDescent(X, y, alpha, iters); 
fprintf('\nThetas:');
theta
fprintf('\nCost:');
cost

%%ps2-3.)
fprintf('\n\nps2-3.');
theta = normalEqn(X, y);

fprintf('\nThetas:');
theta
%%There is a significant difference between the two estimations as 
%%there are only 15 iterations. This low number of iterations is not 
%%sufficient for the gradient decent algorithm to operate. In order to 
%%see similar results between the two, we should increase the number of 
%%iterations.


%%ps2-4.)
%%ps2-4-a.)
data = load('input/hw2_data1.csv');

%%ps2-4-b.)
figure(1);
plot(data(:,1), data(:,2),'rx','markersize',8);         
xlabel('Horse power of a car in 100s','FontSize',8);       
ylabel('Price in $1,000s','FontSize',8);
title('Scatter Plot of Training Data','FontSize',10);

%%ps2-4-c.)
m = length(data(:,1)); %number of training examples
X = [ones(m, 1), data(:,1)]; %add a column of ones to x
fprintf('\n\nps2-4-c.)');
fprintf('\nSize of X:');
size(X)
fprintf('\nSize of Y:');
size(data(:,2))

%%ps2-4-d.)
XY = [X, data(:,2)];%Keeps x and Y values together for shuffeling
shuffledArray = XY(randperm(size(X,1)),:);
trainsize=round(size(X,1)*0.9);
testsize=round(size(X,1)*0.1);
X_train = zeros(trainsize,2); % Size of Train Data
X_test = zeros(testsize,2); % Size of Test Data
Y_train = zeros(trainsize,1); % Size of Train Data
Y_test = zeros(testsize,1); % Size of Test Data
for i = 1:trainsize
    X_train(i,1:2) = shuffledArray(i,1:2);
    Y_train(i,1) = shuffledArray(i,3);
end

j=1;
for i = trainsize+1:size(X,1)
    X_test(j,:) = shuffledArray(i,1:2);
    Y_test(j,1) = shuffledArray(i,3);
    j=j+1;
end

%%ps2-4-e.)
fprintf('\n\nps2-4-e.');
iters = 500;
alpha = 0.3;
     
[theta, cost] = gradientDescent(X_train, Y_train, alpha, iters); 

figure(2);
plot(1:iters, cost);
xlabel('Itteration','FontSize',8);       
ylabel('Cost','FontSize',8);
title('Cost of Training Data Alpha = 0.3','FontSize',10);
fprintf('\nTheta: ');
theta

%%ps2-4-f.)
fprintf('\n\nps2-4-f.');
pred4f = computeCost(X_test, Y_test, theta);
fprintf('\nPredicted Error: %d',pred4f);

%%ps2-4-g.)

fprintf('\n\nps2-4-g.');
theta4g = normalEqn(X_train, Y_train);
pred4g = computeCost(X_test, Y_test, theta4g);
fprintf('\nPredicted Error: %d',pred4g);

%%The difference in error between the two methods is almost negligible.
%%This means that given proper values of alpha and number of iterations,
%%the gradient decent algorithm is a good means of predicting values when
%%supplied with enough training data

%%ps2-4-h.)

%%%%Alpha = 0.001;
iters = 300;
alpha = 0.001;
     
[theta, cost] = gradientDescent(X_train, Y_train, alpha, iters); 

figure(3);
plot(1:iters, cost);
xlabel('Itteration','FontSize',8);       
ylabel('Cost','FontSize',8);
title('Cost of Training Data Alpha = 0.001','FontSize',10);

%%%%Alpha = 0.003;
iters = 300;
alpha = 0.003;
     
[theta, cost] = gradientDescent(X_train, Y_train, alpha, iters); 

figure(4);
plot(1:iters, cost);
xlabel('Itteration','FontSize',8);       
ylabel('Cost','FontSize',8);
title('Cost of Training Data Alpha = 0.003','FontSize',10);

%%%%Alpha = 0.03;
iters = 300;
alpha = 0.03;
     
[theta, cost] = gradientDescent(X_train, Y_train, alpha, iters); 

figure(5);
plot(1:iters, cost);
xlabel('Itteration','FontSize',8);       
ylabel('Cost','FontSize',8);
title('Cost of Training Data Alpha = 0.03','FontSize',10);

%%%%Alpha = 3;
iters = 300;
alpha = 3;
     
[theta, cost] = gradientDescent(X_train, Y_train, alpha, iters); 

figure(6);
plot(1:iters, cost);
xlabel('Itteration','FontSize',8);       
ylabel('Cost','FontSize',8);
title('Cost of Training Data Alpha = 3','FontSize',10);

%%The larger the supplied alpha value, the quicker the cost function is to
%%bottom  out. This behavior  can be seen in alpha .001 to .03. A too small
%%alpha will take too long to see the proper minimization of cost. The
%%outlier in this is when alpha = 3. This causes the function to skyrocket.
%%A too large of an alpha value will overshoot the convergence point.


%%ps2-5-a.)
fprintf('\n\nps2-5-a.');
data2 = load('input/hw2_data2.txt');
mean5 = mean(data2,1);
stdDev5 = std(data2,1);
[rows,cols] = size(data2);
standardX = zeros(rows,2);
for i = 1:rows
    standardX(i,1) = (data2(i,1)-mean5(1,1))/stdDev5(1,1);
    standardX(i,2) = (data2(i,2)-mean5(1,2))/stdDev5(1,2);
end

Y5 = data2(:,3);
X5 = [ones(rows, 1),  standardX]; %add a column of ones to x
fprintf('\nMean feature 1: %d',mean5(1,1));
fprintf('\nMean feature 2: %d',mean5(1,2));
fprintf('\nStandard Deviation feature 1: %d',stdDev5(1,1));
fprintf('\nStandard Deviation feature 2: %d',stdDev5(1,2));
fprintf('\nSize of X:');
size(X5)
fprintf('\nSize of Y:');
size(Y5)

%%ps2-5-b.)
fprintf('\n\nps2-5-b.');

iters = 750;
alpha = 0.01;
     
[theta5, cost5] = gradientDescent(X5, Y5, alpha, iters); 

figure(7);
plot(1:iters, cost5);
xlabel('Itteration','FontSize',8);       
ylabel('Cost','FontSize',8);
title('Cost of Training Data 5-b','FontSize',10);
fprintf('\nTheta Values:');
theta5

%%ps2-5-c.)
fprintf('\n\nps2-5-c.');

Xpredict(1,1) = 1;
Xpredict(2,1) = (1250-mean5(1,1))/stdDev5(1,1);
Xpredict(3,1) = (3-mean5(1,2))/stdDev5(1,2);

yPredict = (Xpredict'*theta5);
fprintf('\nPredicted Price: %d',yPredict);


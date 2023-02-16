%%ps3_Hess_Bryan
clear ; 
close all; 
clc;
%%%%%%%%%%%%%%%%%%%%ps3-1-a.)
fprintf('\n\nps3-1-a.)');
data = load('input/hw3_data1.txt');
[m, n] = size(data); %number of training examples m
n=n-1; %number of features n
X = [ones(m, 1), data(:,1),data(:,2)]; %add a column of ones to x
y = [data(:,3)];
fprintf('\nSize of X: ');
size(X)
fprintf('\nSize of y: ');
size(y)

%%%%%%%%%%%%%%%%%%%%ps3-1-b.)
figure(1); 
hold on;

%Finds where in y the X values are 1 or 0
ad = find(y == 1); 
nonad = find(y == 0);

plot(X(ad, 2), X(ad, 3), 'k+','LineWidth', 2);
plot(X(nonad, 2), X(nonad, 3), 'ko', 'MarkerFaceColor', 'y')

%Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')

hold off;

%%%%%%%%%%%%%%%%%%%%ps3-1-c.)

%Splits data into test data and training data
indicies = randperm(length(X)); 
X_train = X(indicies(1:round(length(indicies)*0.9)),:);
y_train = y(indicies(1:round(length(indicies)*0.9)),:);

X_test = X(indicies((round(length(indicies)*0.9))+1:end),:);
y_test = y(indicies((round(length(indicies)*0.9))+1:end));

%%%%%%%%%%%%%%%%%%%%ps3-1-d.)
figure(2); 
hold on;
z = [-10:10];
gz = sigmoid(z);

plot(z, gz);
xlabel('z')
legend('g(z)')

hold off;

%%%%%%%%%%%%%%%%%%%%ps3-1-e.)
fprintf('\n\nps3-1-e.)');
toyy = [0,1,0,1]';
toytheta = [0, 0, 0]';
toyX=[[1,1,1,1]', [0,0,2,2]', [1,3,0,1]'];
[toyJ, toygrad] = costFunction(toytheta, toyX, toyy);
fprintf('\nCost of toy data: %d', toyJ);

%%%%%%%%%%%%%%%%%%%%ps3-1-f.)
fprintf('\n\nps3-1-f.)');
initial_theta = zeros(n+1, 1); %sets initial thetas to 0
options = optimset('GradObj','on','MaxIter',400); %sets options

%Optimal theta and cost
[traintheta, traincost] = ...
    fminunc(@(t)(costFunction(t, X_train, y_train)), initial_theta, options);
fprintf('\nCost of train data: %d', traincost);
fprintf('\nThetas');
traintheta

%%%%%%%%%%%%%%%%%%%%ps3-1-g.)
figure(3);

%Finds where in y the X values are 1 or 0
ad = find(y == 1); 
nonad = find(y == 0);

hold on;

plot(X(ad, 2), X(ad, 3), 'k+','LineWidth', 2);
plot(X(nonad, 2), X(nonad, 3), 'ko', 'MarkerFaceColor', 'y')

%Only need 2 endpoints to define a line
minNmax_x = [min(X(:,2)), max(X(:,2))];

%Calculate the decision line (y=mx+b)
plot_y = (-1/traintheta(3))*(traintheta(2)*minNmax_x+traintheta(1));

plot(minNmax_x, plot_y)

%Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
axis([30, 100, 30, 100])
hold off;

%%%%%%%%%%%%%%%%%%%%ps3-1-h.)
fprintf('\n\nps3-1-h.)');

predict = sigmoid(X_test * traintheta)>=0.5 ;
accuracy = mean(double(predict==y_test))*100;
fprintf('Accuracy: %f', accuracy);

%%%%%%%%%%%%%%%%%%%%ps3-1-i.)
fprintf('\n\nps3-1-i.)');
prob = sigmoid([1 50 75] * traintheta);
fprintf('\nProbability of admission: %d', prob);
if prob>0.5
    fprintf('\nAdmitted');
elseif prob<0.5
    fprintf('\nNot Admitted');
end

%%%%%%%%%%%%%%%%%%%%ps3-2-a.)
fprintf('\n\nps3-2-a.)');
load('input/hw3_data2.mat');
[m, n] = size(data); %Number of training examples m
n=n-1;
population = [ones(m, 1), data(:,1), data(:,1).*data(:,1)]; %Add a column of ones to x
profit = [data(:,2)];

profitthetas = normalEqn(population, profit);

fprintf('\nThetas');
profitthetas

%%%%%%%%%%%%%%%%%%%%ps3-2-b.)
fprintf('\n\nps3-2-b.)');

figure(4); 
hold on
fittedmodel = profitthetas(1) + profitthetas(2).*population(:,2) + profitthetas(3).*population(:,2).*population(:,2); %Fit line

plot(sort(population(:,2)), sort(fittedmodel), 'b');
plot(population(:,2), profit, 'or');

%Labels and Legend
xlabel('population in thousands, n')
ylabel('profit')
legend('fitted model', 'training data')
hold off


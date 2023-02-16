clear all;

%% 1. Naïve-Bayes classifier

%Splits data into test data and training data
data = csvread('input/diabetes.csv', 1);

[m, n] = size(data); %number of training examples m
n=n-1; %number of features n
X = [data(:,1),data(:,2),data(:,3),data(:,4),data(:,5),data(:,6),data(:,7),data(:,8)];
y = [data(:,9)];

indicies = randperm(length(X)); 
X_train = X(indicies(1:round(length(indicies)*double(540/768))),:);
y_train = y(indicies(1:round(length(indicies)*double(540/768))),:);

X_test = X(indicies((round(length(indicies)*double(540/768)))+1:end),:);
y_test = y(indicies((round(length(indicies)*double(540/768)))+1:end));

%%1.a
X_train_0 = zeros(1, 8);
X_train_1 = zeros(1, 8);
c0 = 1;
c1 = 1;

for i=1:size(X_train, 1)
  if(y_train(i) == 0)
    X_train_0(c0, :) = X_train(i, :);
    c0 = c0 + 1;
  else 
    X_train_1(c1, :) = X_train(i, :);
    c1 = c1 + 1;
  end
end

%%1.b
mean_0 = mean(X_train_0)';
stdev_0 = std(X_train_0)';

mean_1 = mean(X_train_1)';
stdev_1 = std(X_train_1)';

t = table(mean_0, mean_1, stdev_0, stdev_1, 'VariableNames', {'Class 0 Mean', 'Class 0 STDEV', 'Class 1 Mean', 'Class 1 STDEV'});
t


prob0 = zeros(size(X_test, 1), 8);
prob1 = zeros(size(X_test, 1), 8);

%%1.c
for i = 1:size(X_test, 1)
  for j = 1:8
    prob0(i,j) = sqrt(2*pi*stdev_0(j))*exp(-((X_test(i,j)-mean_0(j))^2)/(2*stdev_0(j)));
    prob1(i,j) = sqrt(2*pi*stdev_1(j))*exp(-((X_test(i,j)-mean_1(j))^2)/(2*stdev_1(j)));
  end
end


postProb = zeros(size(X_test, 1));

postProb(:, 1) = prod(prob0, 2) * 0.65;
postProb(:, 2) = prod(prob1, 2) * 0.35;

class = zeros(size(postProb, 1), 1);

for i=1:size(postProb, 1)
  if(postProb(i, 1)) > postProb(i, 2)
    class(i) = 0;
  else
    class(i) = 1;
  end
end

fprintf('Accuracy is: %.4f\n',100*(size(y_test)-sum(abs(y_test-class)))/size(y_test));


%% 2. Minimum distance classifier 

%%1.a
C = cov(X_train);
C

%%1.b 
%%Already did that in 1.b see mean_0 and mean_1

%%1.c
mClass = zeros(size(X_test, 1), 1);
for i=1:size(X_test, 1)
  d0 = ((X_test(i,:)-mean_0')*(C.^-1)*(X_test(i,:)-mean_0')')^(0.5);
  d1 = ((X_test(i,:)-mean_1')*(C.^-1)*(X_test(i,:)-mean_1')')^(0.5);

  d0T = d0-log(0.65);
  d1T = d1-log(0.35);

  if(d0T < d1T)
    mClass(i) = 0;
  else
    mClass(i) = 1;
  end
end

fprintf('Accuracy is: %.4f\n',100*(size(y_test)-sum(abs(y_test-mClass)))/size(y_test));


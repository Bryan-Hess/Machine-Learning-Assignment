function P = predict(theta1, theta2, x)

P = zeros(size(x,1), 1);
a = ones(size(x,1), 1);
x = [a x];
z1 = x*theta1';
a2 = sigmoid(z1);
a2 = [a a2];
z2 = a2*theta2';
a3 = sigmoid(z2);

%Index of column with max value
[~, P] = max(a3, [], 2);


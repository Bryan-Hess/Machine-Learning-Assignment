function [J, grad] = costFunction(theta, X, y)
  m = length(y); %number of samples
  J = (1/m) * sum(-y'*log(sigmoid(X*theta)) - (1-y)'*log(1-sigmoid(X*theta)));
  grad = (1/m) * sum( X.*repmat((sigmoid(X*theta) - y), 1, size(X,2))); 
end
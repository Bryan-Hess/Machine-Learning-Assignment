 function [theta, cost] = gradientDescent(X_train, y_train, alpha, iters)
     cost = zeros(iters, 1);
     [m,nP1]=size(X_train); %number of features and training examples
     theta = rand(nP1,1); %randomizes theta
     for i = 1:iters
        %Parce through each feature
        for j = 1:nP1
            h = theta' * X_train';
            next = (h' - y_train).*X_train(:,j);
            theta(j) = theta(j) - (alpha/m)*sum(next); %Theta(i)
        end
        cost(i) = computeCost(X_train, y_train, theta);
     end
 end
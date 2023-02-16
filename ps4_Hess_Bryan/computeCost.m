function [J] = computeCost(X, y, theta)
    [m,n] = size(X); %Get size constraints (only need m)
    h = X * theta; %Calculate h(x)i
    J = sum((h-y).^2)/(2*m); %Return cost
end
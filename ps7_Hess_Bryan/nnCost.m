%%ps7-2-a.)

function J = nnCost(Theta1, Theta2, X, y, K, lambda)

    m = size(X, 1);

    %z and h(a) for each layer
    a1 = [ones(m,1) X];
    z1 = a1*Theta1';
    a2 = sigmoid(z1);
    a2 = [ones(m,1) a2];
    z2 = a2*Theta2';
    a3 = sigmoid(z2);
    
    %Reform y into 1s and 0s
    identK = eye(K);
    yRecoded = identK(y,:);
    
    %Cost of J
    cost = (-1/m).*sum(sum(yRecoded.*log(a3)+(1-yRecoded).*log(1-a3),2));
    
    %Regularization of J
    reg = (lambda/(2*m)).*( sum(sum(Theta1(:,:).^2))+ sum(sum(Theta2(:,:).^2)));
        
    J = cost+reg;

end


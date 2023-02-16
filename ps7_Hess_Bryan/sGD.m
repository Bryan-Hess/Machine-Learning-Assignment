%% 4. Backpropagation for Gradient of Cost Functions and Stochastic Gradient Descent
function [Theta1, Theta2] = sGD(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda, alpha, MaxEpochs)

    m = size(X_train, 1);
    
    %%ps7-4-a.)
    %Theta 1 and 2 generated randomly on the interval [-0.1 0.1]
    Theta1 = -(-0.1)+(0.1-(-0.1)).*rand(hidden_layer_size,input_layer_size+1);
    Theta2 = -(-0.1)+(0.1-(-0.1)).*rand(num_labels,hidden_layer_size+1);
    
    %Reform y into 1s and 0s
    identNumLab = eye(num_labels);
    yRecoded = identNumLab(y_train,:);
    
    for epoch = 1:MaxEpochs
        %%ps7-4-b.)
        %Forward pass
        a1 = [ones(m,1) X_train];
        z2 = a1* Theta1';
        a2 = sigmoid(z2);
        a2 = [ones(m,1) a2];
        z3 = a2 * Theta2';
        a3 = sigmoid(z3);

        %Error
        direct3 = a3-yRecoded;
        direct3Theta2 = (direct3*Theta2);    
        direct2 = direct3Theta2(:,2:end).*sigmoidGradient(z2);

        %%ps7-4-c.)
        %Prevent first col from regularization
        Theta1 = [zeros(hidden_layer_size,1) Theta1(:,2:end)];
        Theta2 = [zeros(num_labels,1) Theta2(:,2:end)];

        %Add regularization to the gradient
        D1 = (direct2'*a1)+(lambda*Theta1);
        D2 = (direct3'*a2)+(lambda*Theta2);
        
        %%ps7-4-d.)
        %Update thetas
        Theta1 = Theta1-(alpha*D1);
        Theta2 = Theta2-(alpha*D2);
        
        %Find the cost of the new thetas
        J = nnCost(Theta1, Theta2, X_train, y_train, num_labels, lambda);
        
        %ps7-4-e.)
        %If error < 10^-4, break
        if(epoch~=1 && abs(J - prevJ) < 1e-4)
            break;
        end
        prevJ = J;
    end
        

        
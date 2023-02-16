function theta = Reg_normalEqn(X_train,y_train,lambda)
    [m, nP1] = size(X_train);
    theta = pinv(X_train'*X_train + lambda.*eye(nP1))*(X_train'*y_train);
end
function [y_predict] = weightedKNN(X_train, y_train, X_test, sigma)
    distance = pdist2(X_test, X_train);
    mag = -(distance).^2;
    weight = exp(mag/(sigma^2));
    y_predict = zeros(size(X_test,1),1);
    
    for i = 1:size(X_test,1)
        for j = 1:size(y_train,1)
            indicie = find(y_train == j);
            weightSum(j) = sum(weight(i,indicie), 'all');
        end
        %If wieghts are all insignificant, then chooses maximum from the
        %bunch, hence the max
        y_predict(i) = max(find(weightSum == max(weightSum)));
    end
    
end
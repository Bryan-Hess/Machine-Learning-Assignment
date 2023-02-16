%%ps9-1-a.)
function [ids, means, ssd] = kmeans_single(X, k, iters)
    [m,n] = size(X);
    minX = min(X);
    rangeFeatures = abs(max(X) - minX);

    %Initialize means
    for c = 1:k
        for j = 1:n
            random(j) = rangeFeatures(j)*rand(1)+minX(j);
        end
        means(c,:) = random;
    end
        
    for i = 1:iters %Loop Itteration
        dist = pdist2(X,means); %Distance between image and means
        [~, ids] = min(dist, [], 2); %Index of min dist
        for c = 1:k %Loop clusters
            sum = zeros(1,n);
            cnt = 0;
            for j = 1:m %Loop m
                if(ids(j) == c)
                    sum = sum + X(j,:);
                    cnt = cnt+1;
                end
            end
            means(c,:) = sum/cnt; %Avg of each col
        end
    end
    dist = pdist2(X,means);
    temp = 0;
    for index_ssd = 1:m %Loop m
        temp = temp + dist(index_ssd, ids(index_ssd))^2;
    end
    ssd = temp;
end


%%ps9-1-b.)
function [ids, means, ssd] = kmeans_multiple(X, K, iters, R)
    for res = 1:R
        [All_ids{res}, All_means{res}, All_ssds(res)] = kmeans_single(X, K, iters);
    end
    
    %Means/ids for lowest ssd
    [ssd, index] = min(All_ssds);
    means = All_means{index};
    ids = All_ids{index};
end


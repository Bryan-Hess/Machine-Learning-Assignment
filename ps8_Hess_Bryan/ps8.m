%% ps8_Hess_Bryan
clear all;
load('./input/HW8_data1.mat');

%% Part 1: Bagging and Handwritten-digits classification
%%ps8-1-a.)
%Pick 25 Images
indicies = randperm(length(X)); %gives random permutation of all row indicies in feature matrix
randomImages = X(indicies(1:25),:);

%Display
figure(1);
for i = 1:size(randomImages, 1) 
   subplot(5, 5, i);
   imshow(reshape(randomImages(i,:), [20 20]));

end

%%ps7-1-b.)
%%Split data into test and train
X_train = X(indicies(1:round(length(indicies)*0.9)), :);
y_train = y(indicies(1:round(length(indicies)*0.9)), :);

X_test = X(indicies((round(length(indicies)*0.9))+1:end),:);
y_test = y(indicies((round(length(indicies)*0.9))+1:end));

%%ps8-1-c.)
%Number of bags and size of each bag
numOfBags = 5;
bagSize = size(X_train,1)/numOfBags;

for i = 1:numOfBags
    %Sets up subsets
    subsets_X = cell(1,numOfBags);
    subsets_y = cell(1,numOfBags);
    
    index = randperm(size(X_train,1));
    j = 1; %Counter for subsets

    for k = 1:numOfBags      
        %Pulls and stores X and Y value
        subsets_X{j} = X_train(index((bagSize*k)-(bagSize-1):(bagSize*k)),:);
        subsets_y{j} = y_train(index((bagSize*k)-(bagSize-1):(bagSize*k)),:);
        j = j + 1;   
    end
end

%Creates X1-X5 and stores them
[X1,X2,X3,X4,X5] = subsets_X{1,1:5};
[Y1,Y2,Y3,Y4,Y5] = subsets_y{1,1:5};

save('./input/X1.mat','X1');
save('./input/X2.mat','X2');
save('./input/X3.mat','X3');
save('./input/X4.mat','X4');
save('./input/X5.mat','X5');
save('./input/Y1.mat','Y1');
save('./input/Y2.mat','Y2');
save('./input/Y3.mat','Y3');
save('./input/Y4.mat','Y4');
save('./input/Y5.mat','Y5');

%%ps8-1-d.)
%OneVOne = fitcecoc(X1, Y1, 'Learners', templateSVM('BoxConstraint', 0.1));
OneVOne = fitcecoc(X1, Y1);
SVMOneVsOneError=zeros(1,6);
for i=1:5
    SVMOneVsOneError(i) = loss(OneVOne,subsets_X{1,i},subsets_y{1,i});
end
SVMOneVsOneError(6) = loss(OneVOne,X_test,y_test);
SVMOneVsOneError
OneVOnePred=predict(OneVOne,X_test);
%t = table(SVMOneVsOneError, 'VariableNames', {'X1vsX1', 'X1vsX2', 'X1vsX3', 'X1vsX4', 'X1vsX5', 'X1vsX_Test'})

%%ps8-1-e.)
tTree = templateTree('surrogate','on');
tEnsemble = templateEnsemble('GentleBoost',100,tTree);

options = statset('UseParallel',true);
OneVsAll = fitcecoc(X2,Y2,'Coding','onevsall','Learners',tEnsemble,...
                'Prior','uniform','NumBins',50,'Options',options);
SVMOneVsAllError=zeros(1,6);
for i=1:5
    SVMOneVsAllError(i) = loss(OneVsAll,subsets_X{1,i},subsets_y{1,i});
end
SVMOneVsAllError(6) = loss(OneVsAll,X_test,y_test);
SVMOneVsAllError
OneVsAllPred=predict(OneVsAll,X_test);


%%ps8-1-f.)
X3_noise = awgn(X3,10);

NaiveBayes = fitcnb(X3_noise, Y3);
NaiveBayesError=zeros(1,6);
for i=1:5
    NaiveBayesError(i) = loss(NaiveBayes,subsets_X{1,i},subsets_y{1,i});
end
NaiveBayesError(6) = loss(NaiveBayes,X_test,y_test);
NaiveBayesError
NaiveBayesPred=predict(NaiveBayes,X_test);

%%ps8-1-g.)
TreeClass = fitctree(X4, Y4);
TreeClassError=zeros(1,6);
for i=1:5
    TreeClassError(i) = loss(TreeClass,subsets_X{1,i},subsets_y{1,i});
end
TreeClassError(6) = loss(TreeClass,X_test,y_test);
TreeClassError
TreeClassPred=predict(TreeClass,X_test);

%%ps8-1-h.)
TreeBag = TreeBagger(80, X5, Y5);
TreeBagError=zeros(1,6);

pred1=str2double(predict(TreeBag,X1));
pred2=str2double(predict(TreeBag,X2));
pred3=str2double(predict(TreeBag,X3));
pred4=str2double(predict(TreeBag,X4));
pred5=str2double(predict(TreeBag,X5));
TreeBagPred=str2double(predict(TreeBag,X_test));

err1 = find(pred1~=Y1);
err2 = find(pred2~=Y2);
err3 = find(pred3~=Y3);
err4 = find(pred4~=Y4);
err5 = find(pred5~=Y5);
err6 = find(TreeBagPred~=y_test);

TreeBagError(1) = size(err1,1)/size(Y1,1);
TreeBagError(2) = size(err2,1)/size(Y2,1);
TreeBagError(3) = size(err3,1)/size(Y3,1);
TreeBagError(4) = size(err4,1)/size(Y4,1);
TreeBagError(5) = size(err5,1)/size(Y5,1);
TreeBagError(6) = size(err6,1)/size(y_test,1);

TreeBagError

%%ps8-1-i.)
majorityVote = zeros(size(X_test,1), 1);

%Loop through X_test
for i = 1:size(X_test, 1)
    %All predictions
    majVotePred = [OneVOnePred(i) OneVsAllPred(i) NaiveBayesPred(i) TreeClassPred(i) TreeBagPred(i)];
    
    %Majority vote
    majorityVote(i) = mode(majVotePred);
end

majErr = find(majorityVote~=y_test);
MajorityVoteError = size(majErr,1)/size(y_test,1)




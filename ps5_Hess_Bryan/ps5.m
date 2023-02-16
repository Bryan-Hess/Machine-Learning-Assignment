clear all;

%% 0.  Data Preprocessing 
dataPreProcessing();

images = dir(strcat("input/train/", "*.pgm"));
T = zeros(10304, 320);
firstFace = imread(strcat("input/train/", images(1).name));
figure(7)
imshow(firstFace, [])

%% 1. PCA analysis 

%%1.a
for i = 1:size(images)
  trainImg = imread(strcat("input/train/", images(i).name));
  T(:, i) = trainImg(:);
end

figure(1)
imshow(T, [])

%%1.b
meanOfImages = mean(T, 2);
avgFace = reshape(meanOfImages, 112, 92);

figure(2)
imshow(avgFace, [])

%%1.c
A = T - meanOfImages;
C = A*A';

figure(3)
imshow(C, []);

%%1.d

eigenvalues = sort(eig(A'*A), 'DESCEND');
var = zeros(1, 320);

for k = 1:320
    numer = 0;
    denom = sum(eigenvalues, 'all');

    for j = 1:k
       numer = numer + eigenvalues(j, 1);
    end

    var(k) = numer/denom;
end
kplot = 1:1:320;

figure(4)
plot(kplot, var);
title('k vs v(k)')
xlabel('k')
ylabel('v(k)')

%%1.e
[U,D] = eigs(C,162);
temp = U(:,1:8);
eightFaces = zeros(112,92,8);
for i = 1:8
    eightFaces(:,:,i) = reshape(temp(:,i),[112,92]);
end

figure(5);
for i = 1:8
    subplot(2,4,i);
    imshow(eightFaces(:,:,i), []);
end

disp("The size of U is: ");
disp(size(U));
saveas(gcf, 'output/ps5-1-e.png');

%% 2. Feature extraction for face recognition
%%2.a

W_training = zeros(320, 162);
labels_training = zeros(320, 1);

for i=1:320
    %%Similar to as in 1.a
    trainImg = imread(strcat("input/train/", images(i).name));
	W_training(i, :) = U'*(double(trainImg(:)) - meanOfImages);
    
    %%Keeps label
	tmp = split(images(i).name,'-');
	labels_training(i) = str2num(string(tmp(1)));
end
disp("Size of W_training");
disp(size(W_training))

%%2.b
W_testing = zeros(80, 162);
labels_testing = zeros(80, 1);

cnt=1;
%%Loops through all test directories
for(i=1:40)
    %%Pulls image from directory
    thisDir = strcat("input/test/s", string(i), '/');
    images = dir(strcat(thisDir, "*.pgm"));
    
    %%Pulls first test image
    testImg = imread(strcat(thisDir, images(1).name));
    labels_testing(cnt) = i;
    W_testing(cnt, :) = U'*(double(testImg(:)) - meanOfImages); %typecast to double as array is uint8
    cnt = cnt+1;
    
    %%Pulls second test image
    testImg = imread(strcat(thisDir, images(2).name));
    labels_testing(cnt) = i;
    W_testing(cnt, :) = U'*(double(testImg(:)) - meanOfImages); %typecast to double as array is uint8
    cnt=cnt+1;
end
disp("Size of W_testing");
disp(size(W_testing))

%% 3.  Face recognition 
%%3.a

k = 1:2:11;
knn_acc = zeros(6, 1);

%%Tests all k's
for i = 1:6
  %%knn function
  knn = fitcknn(W_training, labels_training, 'NumNeighbors', k(i));
  
  %%modified from ps4, accuracy increases per accurate prediction
  for j=1:80
      [label, score, cost] = predict(knn, W_testing(j, :));
      
      if(label == labels_testing(j))
        knn_acc(i) = knn_acc(i) + 1;
      end
  end
  knn_acc(i) = knn_acc(i)/80;
end

knnTable = table(k', knn_acc,'VariableNames', {'k Value' 'Accuracy'});
disp(knnTable)

%%3.b SOMETHING DOES NOT ALLOW FOR THIS PART OF THE CODE TO PROPERLY COMPUTE ACCURACIES
svm_accLin = zeros(1, 3);
svm_accPoly = zeros(1, 3);
svm_accRBF = zeros(1, 3);

for i=1:40
  svmLabels_training = zeros(320, 1);
  for j=1:8
    svmLabels_training((i-1)*8+j) = i;
  end
  svm_classLin{i} = fitcsvm(W_training, svmLabels_training, 'ClassNames', [false true], 'Standardize', true, 'KernelFunction', 'Linear', 'BoxConstraint', 1);
  svm_classPoly{i} = fitcsvm(W_training, svmLabels_training, 'ClassNames', [false true], 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3);
  svm_classRBF{i} = fitcsvm(W_training, svmLabels_training, 'ClassNames', [false true], 'Standardize', true, 'KernelFunction', 'RBF');
end

svm_predLin = zeros(80, 1);
svm_predPoly = zeros(80, 1);
svm_predRBF = zeros(80, 1);

for i=1:80
  top1 = 0;
  top2 = 0;
  top3 = 0;
  for j=1:40
    [label, score] = predict(svm_classLin{j}, W_testing(i, :));
    if(score(1) > top1)
      svm_predLin(i) = j;
    end
  end
  if(labels_testing(i) == svm_predLin(i))
    svm_accLin(1) = svm_accLin(1) + 1;
  end
    [label, score] = predict(svm_classPoly{j}, W_testing(i, :));
    if(score(1) > top2)
      svm_predPoly(i) = j;
    end
  
  if(labels_testing(i) == svm_predPoly(i))
    svm_accPoly(1) = svm_accPoly(1) + 1;
  end
    [label, score] = predict(svm_classRBF{j}, W_testing(i, :));
    if(score(1) > top3)
      svm_predRBF(i) = j;
    end
  
  if(labels_testing(i) == svm_predRBF(i))
    svm_accRBF(1) = svm_accRBF(1) + 1;
  end
end

svm_accLin(1) = svm_accLin(1) / 80;
svm_accPoly(1) = svm_accPoly(1) / 80;
svm_accRBF(1) = svm_accRBF(1) / 80;

%%svmTable = table(svm_class, svm_acc,'VariableNames', {'SVM_class' 'Accuracy'});
%%disp(svmTable)



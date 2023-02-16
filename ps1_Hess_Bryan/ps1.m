%%ps1-3-a.)
x = 1.5 + 0.6.*randn(1000000,1,'double');

%%ps1-3-b.)
z = (2-(-2)).*rand(1000000,1) - 2;

%%ps1-3-c.)
figure(1);
histogram(x)
figure(2);
histogram(z)

%%ps1-3-d.)
tic
for i = 1:size(x)
       x(i)=x(i)+1;
end
toc

%%ps1-3-e.)
tic
    x = x+1;
elapsedTime = toc;
fprintf('Elapsed time for second opperation is %d seconds', elapsedTime);
%%Adding to each element wihout a loop is much faster than using a loop

%%ps1-3-f.)
y=[];
for i = 1:size(z)
       if (0<=z(i))&&(z(i)<0.5)
           y(end+1)=z(i);
       end
end
y=y';

%%y comes out to have ≈1/8 the number of elements of z, which makes sense
%%as all positive numbers below 0.5 is 1/8 of the range of the normal
%%distibution z. The number retrieved is slighlty different each run as z
%%isn't a perfect normal distibution

%%ps1-4-a.)
A = [2 1 3; 2 6 8; 6 8 18];

%minimum of each row
fprintf('\nMinimum of each column: ');
min(A,[],1)
%minimum of each column
fprintf('\nMaximum of each row: ');
max(A,[],2)
%minimum of whole array
fprintf('\nMinimum of whole array: ');
min(A,[],'all')
%sum of each row
fprintf('\nSum of each row: ');
sum(A,2)
%sum of whole array
fprintf('\nSum of whole array: ');
sum(A,'all')
%square array elements
fprintf('\nSquare of array elements: ');
B = A.^2

%%ps1-4-b.)
syms b n m;
eqn1 = 2*b + n + 3*m == 1;
eqn2 = 2*b + 6*n + 8*m == 3;
eqn3 = 6*b + 8*n + 18*m == 5;
[O,P] = equationsToMatrix([eqn1, eqn2, eqn3], [b, n, m]);
X = linsolve(O,P)

%%ps1-4-c.)
x1 = [0.5 0 -1.5];
x2 = [1 -1 0];

%%L1 norm
fprintf('\nL1 norm for x1: ');
norm(x1,1)
fprintf('\nL1 norm for x2: ');
norm(x2,1)
%%L2 norm
fprintf('\nL2 norm for x1: ');
norm(x1)
fprintf('\nL2 norm for x2: ');
norm(x2)

%%ps1-5.)
mat1 = [1,2,3;4,5,6];
mat2 = [1,2;3,4;5,6];

normal1 = normalize_col(mat1)
normal2 = normalize_col(mat2)

function [B] = normalize_col(A)
    [NumRows, NumCols]=size(A);
    for i = 1:NumCols
       B(:,i) = A(:,i)./sqrt(sum(A(:,i).^2)); 
    end
end

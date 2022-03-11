function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
parameters = [0.01 0.03 0.1 0.3 1 3 10 30];

A = zeros(64,3);
index_row = 1;
len_num = length(parameters);

for i = 1:len_num
  C = parameters(i);
  for j = 1:len_num
    sigma = parameters(j);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    cross_val_error = mean(double(predictions ~= yval));

    A(index_row,:) = [C, sigma, cross_val_error];
    index_row = index_row + 1;
  end
end

[min_value, index] = min(A(:, 3));

C = A(index,1);
sigma = A(index,2);

end

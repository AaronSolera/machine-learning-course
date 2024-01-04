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

iter = 3;
c_sigma_list = zeros(iter^2, 2);
error_list = zeros(iter^2, 1);

printf("Computing optimal C and sigma values\n");
for c = 1:iter
    for s = 1:iter
        printf("Iteration: %i\n", (c-1)*iter+s);
        C = c_sigma_list((c-1)*iter+s, 1) = 0.01 * 10^c;
        sigam = c_sigma_list((c-1)*iter+s, 2) = 0.01 * 10^s;
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        error_list((c-1)*iter+s) = mean(double(svmPredict(model, Xval) ~= yval));
    end
end

[_ idx] = min(error_list);
C = c_sigma_list(idx, 1);
sigma = c_sigma_list(idx, 2);
% =========================================================================

end

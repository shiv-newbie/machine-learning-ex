function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% =============================================================

% The first term of J is the same as without regualrization
J = -1 .* (log(sigmoid(X*theta))' * y + log(1 - sigmoid(X*theta))' * (1 - y))/m;
temp = 0;
% The second term should be summation over the second term till the last
for i = 2:size(theta)
    temp = temp + (theta(i))^2 * lambda/(2*m);
end
J = J + temp;

% =============================================================

% The first term of gradient is the same for all theta as without regualrization
grad = X' * (sigmoid(X*theta) - y)/m;
% The second term should be summation over the second term till the last
for i = 2:size(theta)
    grad(i) = grad(i) + lambda * theta(i)/m;
end

% =============================================================

end

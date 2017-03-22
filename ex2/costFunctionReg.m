function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1 : m,
	z = X(i,:) * theta;
	hepo = sigmoid(z);
	J = J + (-y(i))*log(hepo) - (1-y(i))*log(1-hepo);
	grad = grad + (hepo-y(i))*X(i,:)';
end
J = J / m;

pen = 0;
for i = 2 : size(X, 2),
	pen = pen + theta(i,1)^2;
	grad(i,1) = grad(i,1) + lambda*theta(i,1);
end
J = J + lambda * pen / 2 / m;
grad = grad ./ m;




% =============================================================

end

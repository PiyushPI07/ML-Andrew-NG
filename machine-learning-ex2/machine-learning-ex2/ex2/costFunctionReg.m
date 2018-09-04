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
h=sigmoid(X*theta);
  tmp=-y.*log(h)-(1-y).*log(1-h);
S=sum(tmp(2:m,:));
J1=S/m;
J2=J1+(lambda)/(2*m)*sum(theta(2:length(theta),:).^2);
J=J2+tmp(1,1)/m;


temp=h-y;
G=(temp' *X)';
gradtemp=G/m;
grad=gradtemp+(lambda/(m))*theta;
grad(1,1)=gradtemp(1,1);








% =============================================================

end

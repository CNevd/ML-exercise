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
sum=0;
n=size(theta);

for i=1:m
    htheta=sigmoid(X(i,:)*theta);
    J=J+(-y(i)*log(htheta)-(1-y(i))*log(1-htheta));
    
    grad(1)=grad(1)+(htheta-y(i))*X(i,1);
    for k=2:n
        grad(k)=grad(k)+(htheta-y(i))*X(i,k);
    end
end

grad(1)=grad(1)/m;
for j=2:n
    sum=sum+theta(j)*theta(j);
    grad(j)=grad(j)/m+lambda*theta(j)/m;
end
sum=sum*lambda/2/m;
J=J/m;
J=J+sum;








% =============================================================

end

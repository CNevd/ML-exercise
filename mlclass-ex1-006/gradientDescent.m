function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    add1=0;add2=0;
    aa=(X*theta-y);
    for i=1:m
        add1=add1+aa(i)*X(i,1);
    end
    temp0=theta(1)-alpha/m*add1;
for i=1:m
        add2=add2+aa(i)*X(i,2);
    end
temp1=theta(2)-alpha/m*add2;

theta(1)=temp0;
theta(2)=temp1;




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

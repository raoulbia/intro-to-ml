function [J] = lrCostFunction(theta, X, y, lambda)


m = length(y); % number of training examples

J = 0;

predictions = sigmoid(X * theta) ;

theta_intercept = theta(1) ;
theta_rest = theta(2:end) ;

J = 1/m * ( -y' * log(predictions) - (1-y') * log(1-predictions) ) + ( lambda / (2*m) * sum(theta_rest .^2) ) ;


function unitTest()
X = [
  -15.9368 ;
  -29.1530 ;
   36.1895 ;
   37.4922 ;
  -48.0588 ;
   -8.9415 ;
   15.3078 ;
  -34.7063 ;
    1.3892 ;
  -44.3838 ;
    7.0135 ;
   22.7627
] ;

y =[
    2.1343 ;
    1.1733 ;
   34.3591 ;
   36.8380 ;
    2.8090 ;
    2.1211 ;
   14.7103 ;
    2.6142 ;
    3.7402 ;
    3.7317 ;
    7.6277 ;
   22.7524
] ;


Xval =[

  -16.7465 ;
  -14.5775 ;
   34.5158 ;
  -47.0101 ;
   36.9751 ;
  -40.6861 ;
   -4.4720 ;
   26.5336 ;
  -42.7977 ;
   25.3741 ;
  -31.1096 ;
   27.3118 ;
   -3.2639 ;
   -1.8183 ;
  -40.7197 ;
  -50.0132 ;
  -17.4118 ;
    3.5882 ;
    7.0855 ;
   46.2824 ;
   14.6123 ;
] ;

yval =[

   4.1702e+00 ;
   4.0673e+00 ;
   3.1873e+01 ;
   1.0624e+01 ;
   3.1836e+01 ;
   4.9594e+00 ;
   4.4516e+00 ;
   2.2276e+01 ;
  -4.3874e-05 ;
   2.0504e+01 ;
   3.8583e+00 ;
   1.9365e+01 ;
   4.8838e+00 ;
   1.1097e+01 ;
   7.4617e+00 ;
   1.4769e+00 ;
   2.7192e+00 ;
   1.0927e+01 ;
   8.3487e+00 ;
   5.2782e+01 ;
   1.3357e+01 ;
] ;

m = size(X, 1);
lambda = 0 ;

% get theta using training data
theta = trainLinearReg(X, y, lambda) ; % that function will call linearRegCostFunction

% get cost using
for i=1:m
    theta_train = trainLinearReg(X(1:i, :), y(1:i), lambda) ;

    [J_train, grad_train] = linearRegCostFunction(X(1:i, :), y(1:i), theta_train, lambda) ;
    error_train(i) = J_train ;

    theta_val = trainLinearReg(Xval(1:i, :), yval(1:i), lambda) ;

    [J_val, grad_val] = linearRegCostFunction(Xval(1:i, :), yval(1:i), theta_val, lambda) ;
    error_val(i) = J_val
end


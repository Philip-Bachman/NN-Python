%
% APPROXIMATION FOR GAMMALN (I.E. THE LOG GAMMA FUNCTION)
%
small_approx = @( x, c ) log(1 ./ x) - (0.57721566490153 * x) + (c * x.^2);
large_approx = @( x, c ) (((x-0.5) .* log(x)) - x) + 0.5*log(2*pi) + c*(1./x);

X = linspace(0.01, 4.0, 500);
Y = almost_gammaln(X,0.25,0.025);
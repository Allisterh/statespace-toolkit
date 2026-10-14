function [xi2, acc] = xi2_mh(xi2, SS, A, AA, Lmin, nu, S, step)
% xi2_mh - one sweep of random-walk Metropolis-Hastings on log xi2, one component at a
% time, for the random-walk variances of weights whose prior is the random walk
% theta_1 ~ N(0, Vth*I), theta_t - theta_{t-1} ~ N(0, diag(xi2)), conditioned on the
% restrictions a_t'*theta_t = 1, t = 1, ..., n, with xi2_i ~ IG(nu, S). The full
% conditional of xi2 is the inverse-gamma one of the unrestricted random walk divided by
% p(M*theta = 1 | xi2) = N(1; 0, Q), where Q = Vth*A*A' + (A*diag(xi2)*A').*Lmin is the
% covariance of the n values a_t'*theta_t under the unrestricted random walk.
%   xi2  : 5 x 1 current values
%   SS   : 5 x 1 sums of squared increments of theta
%   A    : n x 5 matrix of the rows a_t'
%   AA   : Vth*A*A'
%   Lmin : n x n matrix with entries min(t,s) - 1
%   step : 5 x 1 proposal standard deviations of log xi2
%   acc  : 5 x 1 indicators of acceptance
n = size(A, 1);
a = nu + (n-1)/2; b = S + SS/2;
v = log(xi2); lc = logpost(v, a, b, A, AA, Lmin);
acc = false(numel(v), 1);
for i = 1:numel(v)
    vp = v; vp(i) = v(i) + step(i)*randn;
    lp = logpost(vp, a, b, A, AA, Lmin);
    if log(rand) < lp - lc
        v = vp; lc = lp; acc(i) = true;
    end
end
xi2 = exp(v);
end

function l = logpost(v, a, b, A, AA, Lmin)
% log full conditional of v = log xi2, up to a constant: the inverse-gamma kernel with
% its Jacobian, minus log N(1; 0, Q)
[C, p] = chol(AA + ((A.*exp(v)')*A').*Lmin, 'lower');
if p > 0, l = -Inf; return; end
u = C\ones(size(A, 1), 1);
l = sum(-a*v - b.*exp(-v)) + sum(log(diag(C))) + .5*(u'*u);
end

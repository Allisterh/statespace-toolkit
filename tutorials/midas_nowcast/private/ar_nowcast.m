function f = ar_nowcast(y, X, xT, nsim, burnin)
% ar_nowcast - Gibbs sampler for the AR(2) benchmark y_t = X(t,:)*beta + e_t,
% e_t ~ N(0, sig2), with X(t,:) = [1, y_{t-1}, y_{t-2}], beta ~ N(0, 10 I) and
% sig2 ~ IG(5, 16), the priors of the constant-parameter MIDAS regressions, and the
% predictive distribution of the target quarter, whose regressors are xT (1 x 3).
%   f.m, f.v : nsim x 1 predictive means and variances
k = size(X, 2);
s2 = var(y);
f.m = zeros(nsim,1); f.v = zeros(nsim,1);
for loop = 1:nsim + burnin
    beta = ssm.simulate_states(eye(k)/10 + X'*X/s2, X'*y/s2);
    e = y - X*beta;
    s2 = 1/gamrnd(5 + numel(y)/2, 1/(16 + e'*e/2));
    if loop > burnin
        f.m(loop-burnin) = xT*beta; f.v(loop-burnin) = s2;
    end
end
end

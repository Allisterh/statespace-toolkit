% ssm.models.tvp_midas - posterior sampler for the MIDAS regressions of Chan, Poon and
% Zhu (2026), whose coefficients, weights and volatility can each vary over time, and the
% nowcast of the next period. Written for this toolkit.
%
%   res = ssm.models.tvp_midas(y, X, W, A)
%   res = ssm.models.tvp_midas(y, X, W, A, 'tvb', true, 'tvtheta', false, 'sv', true, ...
%                              'nsim', 10000, 'burnin', 5000)
%
%   y    : T x 1 low-frequency series
%   X    : regressors with coefficients of their own, such as an intercept, lags of y and
%          monthly predictors, T x k0; a row T+1 holds the regressors of the period to
%          nowcast
%   W, A : 1 x J cells of the regressors w_jt' and the restriction rows a_jt' of the J
%          high-frequency predictors (ssm.midas_basis), each with as many rows as X and
%          the first column of each A{j} equal to 1
%
%   'tvb'     : true lets the coefficients follow random walks; default true
%   'tvtheta' : true lets the weights follow random walks; default true
%   'sv'      : true lets the log-volatility follow a random walk; default true
%   'prior'   : struct with any of the fields Vb and Vtheta, the prior variances of the
%               first or constant coefficients and weights (default 10 each); omega2 and
%               xi2, [nu S] of the IG priors of the random-walk variances of the
%               coefficients and of the weights (defaults [5 .004] and [10 .001]); Vg, the
%               prior variance of g_1 (default 10); sigma2g, [nu S] of the IG prior of the
%               variance of its random walk (default [5 .04]); and sigma2, [nu S] of the IG
%               prior of a constant error variance (default [5 16])
%   'nsim'    : draws kept, default 10000;  'burnin' : draws discarded first, default 5000
%   'seed'    : if given, rng(seed, 'twister') is set before the first draw
%   'thin'    : k > 0 also keeps every kth kept draw of the paths in res.draws; default 0
%   'quantiles' : probabilities of the nowcast quantiles, default [.05 .1 .5 .9 .95]
%
%   res.b_mean     : T x (k0+J) posterior means of the coefficients, those on X first
%   res.theta_mean : 1 x J cell of T x 5 posterior means of the weights
%   res.g_mean     : T x 1 posterior mean of the log-volatility
%   res.max_resid  : largest |a_jt'*theta_jt - 1| over the kept draws, the target period
%                    included; for constant weights, of the restriction averaged over t
%   res.acc        : J x 5 acceptance rates of the steps for the random-walk variances of
%                    the weights, after the burn-in; zeros for constant weights
%   res.nowcast    : with a row to nowcast, m and v (nsim x 1), the mean and variance of
%                    the normal density of y_{T+1} given each draw; mean and q, the mean
%                    and the quantiles of their equally weighted mixture, the predictive
%                    density; probs
%   res.draws      : with thin > 0, b (draws x T x (k0+J)) and theta (1 x J cell of
%                    draws x T x 5)
%   res.prior, and the settings
%
% The model is y_t = x_t'*c_t + sum_j beta_jt*theta_jt'*w_jt + e_t, e_t ~ N(0, exp(g_t)),
% with a_jt'*theta_jt = 1, which identifies beta_jt as the overall impact of predictor j.
% The coefficients b_t = [c_t; beta_t] and each theta_jt start from N(0, Vb*I) and
% N(0, Vtheta*I) and follow random walks with diagonal covariances, whose entries have the
% priors IG(omega2) and IG(xi2); constant ones have the priors N(0, Vb*I) and
% N(0, Vtheta*I), and constant weights meet the restriction averaged over the T periods.
% g_t starts from N(0, Vg) and follows a random walk with an IG(sigma2g) variance; a
% constant error variance has the prior IG(sigma2). IG(nu, S) has density proportional to
% x^(-nu-1)*exp(-S/x). Each sweep draws b given the weights (ssm.simulate_states), the
% weights of each predictor given b and the other weights (ssm.simulate_states, then
% ssm.restrict), g (ssm.ksc_rw_diffuse) and the random-walk variances. The variances of
% the weights are drawn one at a time by random-walk Metropolis-Hastings on their logs,
% with step sizes tuned toward an acceptance rate of 0.44 during the burn-in: their full
% conditional is the inverse-gamma one of the unrestricted random walk divided by
% N(1; 0, Q), with Q = Vtheta*A*A' + (A*diag(xi2)*A').*Lmin and Lmin(t,s) = min(t,s) - 1,
% the covariance of the T values a_t'*theta_t under that random walk. For the period to
% nowcast, each kept draw moves the time-varying states one period on and projects the
% weights onto that period's restriction; these moves use random numbers, so a run with a
% row to nowcast draws other parameter values than the same seed without one.
%
% See:
% Chan, J.C.C., Poon, A. and Zhu, D. (2026). Time-Varying Parameter MIDAS Models:
% Application to Nowcasting US Real GDP, Journal of Econometrics, forthcoming.
% Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated Likelihood
% Estimation in State Space Models, International Journal of Mathematical Modelling and
% Numerical Optimisation, 1(1/2): 101-120.
% Kim, S., Shephard, N. and Chib, S. (1998). Stochastic Volatility: Likelihood Inference
% and Comparison with ARCH Models, Review of Economic Studies, 65(3): 361-393.
% Rue, H. and Held, L. (2005). Gaussian Markov Random Fields: Theory and Applications,
% Chapman & Hall/CRC, Algorithm 2.6.

function res = tvp_midas(y, X, W, A, varargin)
tvb = true; tvtheta = true; sv = true; seed = []; thin = 0;
nsim = 10000; burnin = 5000; probs = [.05 .1 .5 .9 .95];
pr = struct('Vb', 10, 'Vtheta', 10, 'omega2', [5 .004], 'xi2', [10 .001], 'Vg', 10, ...
    'sigma2g', [5 .04], 'sigma2', [5 16]);
iv = 1;
while iv <= numel(varargin)
    if iv == numel(varargin)
        error('ssm:models:tvp_midas:badOption', 'option ''%s'' has no value', char(string(varargin{iv})));
    end
    val = varargin{iv+1};
    switch lower(char(string(varargin{iv})))
        case 'tvb',       tvb = logical(val);
        case 'tvtheta',   tvtheta = logical(val);
        case 'sv',        sv = logical(val);
        case 'prior'
            fn = fieldnames(val);
            for i = 1:numel(fn)
                if ~isfield(pr, fn{i})
                    error('ssm:models:tvp_midas:badOption', 'unknown prior field ''%s''', fn{i});
                end
                pr.(fn{i}) = val.(fn{i});
            end
        case 'nsim',      nsim = val;
        case 'burnin',    burnin = val;
        case 'seed',      seed = val;
        case 'thin',      thin = val;
        case 'quantiles', probs = val;
        otherwise, error('ssm:models:tvp_midas:badOption', 'unknown option ''%s''', char(string(varargin{iv})));
    end
    iv = iv + 2;
end

% checks, and the row to nowcast
y = y(:); n = numel(y);
if ~iscell(W) || ~iscell(A) || numel(W) ~= numel(A)
    error('ssm:models:tvp_midas:badData', 'W and A must be cells with one entry per predictor');
end
J = numel(W); nr = size(X, 1);
if any(~isfinite(y)) || any(~isfinite(X(:)))
    error('ssm:models:tvp_midas:badData', 'y and X must be finite');
end
if nr ~= n && nr ~= n + 1
    error('ssm:models:tvp_midas:badData', 'X has %d rows; it needs T = %d, or T+1 with a row to nowcast', nr, n);
end
kt = 5; if J > 0, kt = size(W{1}, 2); end
for j = 1:J
    if ~isequal(size(W{j}), [nr kt]) || ~isequal(size(A{j}), [nr kt])
        error('ssm:models:tvp_midas:badData', 'W{%d} and A{%d} must be %d x %d, as many rows as X', j, j, nr, kt);
    end
    if any(abs(A{j}(:,1) - 1) > 1e-12) || any(~isfinite([W{j}(:); A{j}(:)]))
        error('ssm:models:tvp_midas:badData', 'A{%d} must have a first column of ones, and W{%d} and A{%d} finite values', j, j, j);
    end
end
if ~(isscalar(nsim) && nsim >= 1 && nsim == fix(nsim) && isscalar(burnin) && burnin >= 0 && burnin == fix(burnin))
    error('ssm:models:tvp_midas:badOption', 'nsim must be a positive integer and burnin a nonnegative integer');
end
nowcast = nr == n + 1;
if nowcast
    xT.x0 = X(n+1,:); xT.w = cell(1,J); xT.a = cell(1,J);
    for j = 1:J
        xT.w{j} = W{j}(n+1,:); xT.a{j} = A{j}(n+1,:);
        W{j} = W{j}(1:n,:); A{j} = A{j}(1:n,:);
    end
    X = X(1:n,:);
end
if ~isempty(seed), rng(seed, 'twister'); end

% priors and the matrices of the random walks
k0 = size(X, 2); kb = k0 + J;
Vb = pr.Vb; Vth = pr.Vtheta; nuom = pr.omega2(1); Som = pr.omega2(2);
nuxi = pr.xi2(1); Sxi = pr.xi2(2); nug = pr.sigma2g(1); Sg = pr.sigma2g(2); Vg = pr.Vg;
nus = pr.sigma2(1); Ss = pr.sigma2(2);
H7 = kron(ssm.diffmat(n), speye(kb)); H5 = kron(ssm.diffmat(n), speye(kt));
Mr = cell(1,J);
for j = 1:J
    if tvtheta
        Mr{j} = sparse(repelem((1:n)', kt), (1:n*kt)', reshape(A{j}', [], 1), n, n*kt);
    else
        Mr{j} = mean(A{j}, 1);
    end
end
if tvtheta
    Lmin = min((0:n-1)', 0:n-1);                % min(t,s) - 1
    AA = cell(1,J);
    for j = 1:J, AA{j} = Vth*(A{j}*A{j}'); end
    step = .3*ones(J, kt); nacc = zeros(J, kt); nbatch = zeros(J, kt);
end

% initial values: theta = e_1 meets every restriction, since the first basis is 1
B = [zeros(n,k0), ones(n,J)];
Th = repmat({[ones(n,1), zeros(n,kt-1)]}, 1, J);
g = log(var(y))*ones(n,1); sig2g = .1;
om2 = .01*ones(kb,1); xi2 = repmat({.001*ones(kt,1)}, 1, J);
ym = zeros(nsim,1); yv = zeros(nsim,1); mres = 0;
b_sum = zeros(n,kb); th_sum = repmat({zeros(n,kt)}, 1, J); g_sum = zeros(n,1);
if thin > 0
    ns = floor(nsim/thin);
    Db = zeros(ns, n, kb); Dth = repmat({zeros(ns, n, kt)}, 1, J);
end

for loop = 1:nsim + burnin
    ie = exp(-g);
    z = zeros(n, J);
    for j = 1:J, z(:,j) = sum(Th{j}.*W{j}, 2); end
    % the coefficients, given the weights
    Xb = [X, z];
    if tvb
        Z = ssm.surform(Xb);
        iS = spdiags([ones(kb,1)/Vb; repmat(1./om2, n-1, 1)], 0, n*kb, n*kb);
        K = H7'*iS*H7 + Z'*spdiags(ie, 0, n, n)*Z;
        B = reshape(ssm.simulate_states(K, Z'*(ie.*y)), kb, n)';
    else
        K = eye(kb)/Vb + Xb'*(ie.*Xb);
        B = repmat(ssm.simulate_states(K, Xb'*(ie.*y))', n, 1);
    end
    % the weights of each predictor, given the coefficients and the other weights
    for j = 1:J
        o = [1:j-1, j+1:J];
        yt = y - sum(X.*B(:,1:k0), 2) - sum(B(:,k0+o).*z(:,o), 2);
        Xt = B(:,k0+j).*W{j};
        if tvtheta
            Zt = ssm.surform(Xt);
            iS = spdiags([ones(kt,1)/Vth; repmat(1./xi2{j}, n-1, 1)], 0, n*kt, n*kt);
            K = H5'*iS*H5 + Zt'*spdiags(ie, 0, n, n)*Zt;
            [th, ~, C] = ssm.simulate_states(K, Zt'*(ie.*yt));
            th = ssm.restrict(th, C, Mr{j}, ones(n,1));
            Th{j} = reshape(th, kt, n)';
        else
            K = eye(kt)/Vth + Xt'*(ie.*Xt);
            [th, ~, C] = ssm.simulate_states(K, Xt'*(ie.*yt));
            th = ssm.restrict(th, C, Mr{j}, 1);
            Th{j} = repmat(th', n, 1);
        end
        z(:,j) = sum(Th{j}.*W{j}, 2);
    end
    % the volatility
    e = y - sum(X.*B(:,1:k0), 2) - sum(B(:,k0+1:end).*z, 2);
    if sv
        g = ssm.ksc_rw_diffuse(log(e.^2 + 1e-4), g, sig2g, Vg);
        sig2g = 1/gamrnd(nug + (n-1)/2, 1/(Sg + sum(diff(g).^2)/2));
    else
        g = log(1/gamrnd(nus + n/2, 1/(Ss + e'*e/2)))*ones(n,1);
    end
    % the random-walk variances
    if tvb
        om2 = 1./gamrnd(nuom + (n-1)/2, 1./(Som + sum(diff(B).^2, 1)'/2));
    end
    if tvtheta
        for j = 1:J
            [xi2{j}, a] = xi2_mh(xi2{j}, sum(diff(Th{j}).^2, 1)', A{j}, AA{j}, Lmin, ...
                nuxi, Sxi, step(j,:)');
            if loop <= burnin
                nbatch(j,:) = nbatch(j,:) + a';
                if mod(loop, 100) == 0          % aim at acceptance rates near 0.44
                    step(j,:) = step(j,:).*exp(nbatch(j,:)/100 - .44);
                    nbatch(j,:) = 0;
                end
            else
                nacc(j,:) = nacc(j,:) + a';
            end
        end
    end
    if loop > burnin
        i = loop - burnin;
        for j = 1:J
            if tvtheta
                mres = max([mres; abs(Mr{j}*reshape(Th{j}', [], 1) - 1)]);
            else
                mres = max(mres, abs(Mr{j}*Th{j}(1,:)' - 1));
            end
        end
        % the period to nowcast: the states one period on, theta projected onto its
        % restriction
        if nowcast
            bT = B(n,:)';
            if tvb, bT = bT + sqrt(om2).*randn(kb,1); end
            mT = xT.x0*bT(1:k0);
            for j = 1:J
                thT = Th{j}(n,:)';
                if tvtheta
                    a = xT.a{j}';
                    thT = thT + sqrt(xi2{j}).*randn(kt,1);
                    thT = thT + xi2{j}.*a*((1 - a'*thT)/(a'*(xi2{j}.*a)));
                    mres = max(mres, abs(xT.a{j}*thT - 1));
                end
                mT = mT + bT(k0+j)*(xT.w{j}*thT);
            end
            if sv, gT = g(n) + sqrt(sig2g)*randn; else, gT = g(n); end
            ym(i) = mT; yv(i) = exp(gT);
        end
        b_sum = b_sum + B; g_sum = g_sum + g;
        for j = 1:J, th_sum{j} = th_sum{j} + Th{j}; end
        if thin > 0 && mod(i, thin) == 0
            Db(i/thin,:,:) = B;
            for j = 1:J, Dth{j}(i/thin,:,:) = Th{j}; end
        end
    end
end

res.b_mean = b_sum/nsim;
res.theta_mean = cellfun(@(s) s/nsim, th_sum, 'UniformOutput', false);
res.g_mean = g_sum/nsim;
res.max_resid = mres;
if tvtheta, res.acc = nacc/nsim; else, res.acc = zeros(J, kt); end
if nowcast
    q = arrayfun(@(p) mixquantile(p, ym, yv), probs);
    res.nowcast = struct('m', ym, 'v', yv, 'mean', mean(ym), 'q', q, 'probs', probs);
end
if thin > 0, res.draws = struct('b', Db, 'theta', {Dth}); end
res.prior = pr;
res.tvb = tvb; res.tvtheta = tvtheta; res.sv = sv;
res.nsim = nsim; res.burnin = burnin; res.seed = seed; res.thin = thin;
end

function [xi2, acc] = xi2_mh(xi2, SS, A, AA, Lmin, nu, S, step)
% one sweep of random-walk Metropolis-Hastings on log xi2, one component at a time;
% SS holds the sums of squared increments of theta, AA = Vtheta*A*A', and step the
% proposal standard deviations
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

function x = mixquantile(p, m, v)
% the p-quantile of the equally weighted mixture of N(m_i, v_i)
s = sqrt(v);
x = fzero(@(q) mean(normcdf((q - m)./s)) - p, [min(m - 8*s) max(m + 8*s)]);
end

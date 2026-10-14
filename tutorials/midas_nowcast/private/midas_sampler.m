function f = midas_sampler(y, X0, W, A, xT, opts, nsim, burnin)
% midas_sampler - Gibbs sampler for the MIDAS regressions of Chan, Poon and Zhu (2026),
% and the predictive distribution of the next quarter:
%   y_t = x0_t'*c_t + sum_j beta_jt*theta_jt'*w_jt + e_t,  e_t ~ N(0, exp(g_t)),
%   a_jt'*theta_jt = 1,  j = 1, ..., J,
% where x0_t holds k0 regressors with their own coefficients, w_jt the basis-weighted
% means of predictor j, and b_t = [c_t; beta_t] and theta_jt (5 each) are random walks
% when opts.tvb and opts.tvtheta are true and constant otherwise; g_t is a random walk
% when opts.sv is true and constant otherwise. A constant theta_j satisfies the mean of
% the rows a_jt. Priors of the paper: b_1 and theta_j1 ~ N(0, 10 I), as are the constant
% coefficients; random-walk variances IG(5, 0.004) for b and IG(10, 0.001) for theta;
% g_1 ~ N(0, 10) and sigma2_g ~ IG(5, 0.04). A constant variance is IG(5, 16).
% b is drawn by ssm.simulate_states, theta by ssm.simulate_states and ssm.restrict, g by
% ssm.ksc_rw_diffuse, and the random-walk variances of theta by xi2_mh.
%   y, X0, W, A : the observed quarters, as rows of nowcast_design (W, A: 1 x J cells of
%                 n x 5 matrices)
%   xT          : the target quarter, with fields x0 (1 x k0), and w and a (1 x J cells of
%                 1 x 5)
%   f.m, f.v    : nsim x 1 predictive means and variances of the target quarter's growth
%   f.res       : the largest restriction residual over the kept draws
%   f.B, f.Th   : with opts.thin > 0, every opts.thin-th kept draw of the paths of b
%                 (draws x n x kb, kb = k0 + J) and of theta (1 x J cell of draws x n x 5)
%   f.acc       : J x 5 acceptance rates of xi2_mh after the burn-in, whose proposal
%                 standard deviations are tuned during the burn-in
n = numel(y); k0 = size(X0, 2); J = numel(W); kb = k0 + J; kt = 5;
Vb = 10; Vth = 10; nuom = 5; Som = .004; nuxi = 10; Sxi = .001;
nug = 5; Sg = .04; Vg = 10; nus = 5; Ss = 16;
H7 = kron(ssm.diffmat(n), speye(kb)); H5 = kron(ssm.diffmat(n), speye(kt));
Mr = cell(1,J);
for j = 1:J
    if opts.tvtheta
        Mr{j} = sparse(repelem((1:n)', kt), (1:n*kt)', reshape(A{j}', [], 1), n, n*kt);
    else
        Mr{j} = mean(A{j}, 1);
    end
end
if opts.tvtheta
    Lmin = min((0:n-1)', 0:n-1);                % min(t,s) - 1
    AA = cell(1,J);
    for j = 1:J, AA{j} = Vth*(A{j}*A{j}'); end
    step = .3*ones(J, kt); nacc = zeros(J, kt); nbatch = zeros(J, kt);
end
f.acc = zeros(J, kt);
% initial values: theta = e_1 satisfies every restriction, since the first basis is 1
B = [zeros(n,k0), ones(n,J)];
Th = repmat({repmat([1 0 0 0 0], n, 1)}, 1, J);
g = log(var(y))*ones(n,1); sig2g = .1;
om2 = .01*ones(kb,1); xi2 = repmat({.001*ones(kt,1)}, 1, J);
f.m = zeros(nsim,1); f.v = zeros(nsim,1); f.res = 0;
thin = 0; if isfield(opts, 'thin'), thin = opts.thin; end
if thin > 0
    ns = floor(nsim/thin);
    f.B = zeros(ns, n, kb); f.Th = repmat({zeros(ns, n, kt)}, 1, J);
end
for loop = 1:nsim + burnin
    ie = exp(-g);
    z = zeros(n, J);
    for j = 1:J, z(:,j) = sum(Th{j}.*W{j}, 2); end
    % the coefficients, given the weights
    Xb = [X0, z];
    if opts.tvb
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
        yt = y - sum(X0.*B(:,1:k0), 2) - sum(B(:,k0+o).*z(:,o), 2);
        Xt = B(:,k0+j).*W{j};
        if opts.tvtheta
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
    e = y - sum(X0.*B(:,1:k0), 2) - sum(B(:,k0+1:end).*z, 2);
    if opts.sv
        g = ssm.ksc_rw_diffuse(log(e.^2 + 1e-4), g, sig2g, Vg);
        sig2g = 1/gamrnd(nug + (n-1)/2, 1/(Sg + sum(diff(g).^2)/2));
    else
        g = log(1/gamrnd(nus + n/2, 1/(Ss + e'*e/2)))*ones(n,1);
    end
    % the random-walk variances
    if opts.tvb
        om2 = 1./gamrnd(nuom + (n-1)/2, 1./(Som + sum(diff(B).^2, 1)'/2));
    end
    if opts.tvtheta
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
    % the target quarter: the states one step on, theta projected onto its restriction
    if loop > burnin
        i = loop - burnin;
        bT = B(n,:)';
        if opts.tvb, bT = bT + sqrt(om2).*randn(kb,1); end
        mT = xT.x0*bT(1:k0);
        for j = 1:J
            thT = Th{j}(n,:)';
            if opts.tvtheta
                a = xT.a{j}';
                thT = thT + sqrt(xi2{j}).*randn(kt,1);
                thT = thT + xi2{j}.*a*((1 - a'*thT)/(a'*(xi2{j}.*a)));
            end
            mT = mT + bT(k0+j)*(xT.w{j}*thT);
            if opts.tvtheta
                f.res = max([f.res; abs(Mr{j}*reshape(Th{j}', [], 1) - 1); abs(xT.a{j}*thT - 1)]);
            else
                f.res = max(f.res, abs(Mr{j}*Th{j}(1,:)' - 1));
            end
        end
        if opts.sv, gT = g(n) + sqrt(sig2g)*randn; else, gT = g(n); end
        f.m(i) = mT; f.v(i) = exp(gT);
        if thin > 0 && mod(i, thin) == 0
            f.B(i/thin,:,:) = B;
            for j = 1:J, f.Th{j}(i/thin,:,:) = Th{j}; end
        end
    end
end
if opts.tvtheta, f.acc = nacc/nsim; end
end

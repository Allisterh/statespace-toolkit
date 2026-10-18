function test_tvp_midas
% ssm.models.tvp_midas on generated data whose windows hold different numbers of
% observations, so that the restriction rows change from period to period. In the five
% configurations of the MIDAS nowcasting tutorial, the restrictions hold in every draw,
% the target period's included, and the posterior means are the means of the kept draws.
% With high-frequency regressors of zeros the data say nothing about the weights, so the
% variances of their random walks keep their prior, which needs the normalizing constant
% of the restricted random walk; and the weights of the target period, with these
% variances held near a value, have the covariance of the random walk conditioned on all
% the restrictions at once. A restriction on the level alone, a_t = e_1, fixes the first
% weight, whose variance then keeps its prior whatever the data. The rows of the 'sum'
% scale of ssm.midas_basis meet their restrictions too; a prior on the variances of the
% weights with one equal row per predictor draws as the shared one; a changed prior
% changes the draws; and malformed input is rejected.

rng(3);
T = 40; n = T + 1;                                   % T quarters and one to nowcast
d = find(rand(30*(n+3), 1) > .3);                     % daily dates with gaps, in days
wend = 30*(4:n+3)'; wstart = wend - 90;               % three-month windows
W = cell(1,2); A = W; Ws = W; As = W;
for j = 1:2
    x = randn(numel(d),1);
    [W{j}, A{j}, m] = ssm.midas_basis(d, x, wstart, wend);
    [Ws{j}, As{j}] = ssm.midas_basis(d, x, wstart, wend, 'scale', 'sum');
end
assert(numel(unique(m)) > 3, 'test_tvp_midas: the windows must hold different numbers of observations');
X = [ones(n,1), randn(n,2)];
y = X(1:T,:)*[.5; .3; -.2] + .8*W{1}(1:T,:)*[1; .2; 0; 0; 0] ...
    - .5*W{2}(1:T,:)*[1; 0; .1; 0; 0] + .5*randn(T,1);

% the five models of the tutorial: [tvb tvtheta sv]
cfg = [0 0 0; 0 0 1; 1 1 1; 1 0 1; 0 1 1];
for c = 1:size(cfg, 1)
    r = ssm.models.tvp_midas(y, X, W, A, 'tvb', cfg(c,1) == 1, 'tvtheta', cfg(c,2) == 1, ...
        'sv', cfg(c,3) == 1, 'nsim', 60, 'burnin', 140, 'seed', 10 + c);
    assert(r.max_resid < 1e-10, 'tvp_midas: the restrictions are not met, model %d', c);
    for j = 1:2
        if cfg(c,2) == 1
            rt = A{j}(n,:)*r.nowcast.theta{j}' - 1;          % the target's restriction
        else
            rt = mean(A{j}, 1)*r.nowcast.theta{j}' - 1;      % the average, the target's row in
        end
        assert(max(abs(rt)) < 1e-10 && isequal(size(r.nowcast.theta{j}), [60 5]), ...
            'tvp_midas: the weights of the target period miss their restriction, model %d', c);
    end
end

% the rows of the 'sum' scale, whose first column counts the observations
r = ssm.models.tvp_midas(y, X, Ws, As, 'nsim', 60, 'burnin', 140, 'seed', 9);
assert(r.max_resid < 1e-10 && max(abs(As{2}(n,:)*r.nowcast.theta{2}' - 1)) < 1e-10, ...
    'tvp_midas: the restrictions of the summed rows are not met');

% one row per predictor of the prior on the variances of the weights
r1 = ssm.models.tvp_midas(y, X, W, A, 'nsim', 20, 'burnin', 10, 'seed', 5);
r2 = ssm.models.tvp_midas(y, X, W, A, 'nsim', 20, 'burnin', 10, 'seed', 5, ...
    'prior', struct('xi2', [10 .001; 10 .001]));
r3 = ssm.models.tvp_midas(y, X, W, A, 'nsim', 20, 'burnin', 10, 'seed', 5, ...
    'prior', struct('xi2', [10 .001; 10 .1]));
assert(isequal(r1.nowcast.m, r2.nowcast.m) && ~isequal(r1.nowcast.m, r3.nowcast.m), ...
    'tvp_midas: the prior of the variances of the weights by predictor does not act as documented');

% the posterior means average the kept draws; no row to nowcast, no nowcast
Wo = cellfun(@(w) w(1:T,:), W, 'UniformOutput', false);
Ao = cellfun(@(a) a(1:T,:), A, 'UniformOutput', false);
r = ssm.models.tvp_midas(y, X(1:T,:), Wo, Ao, 'nsim', 30, 'burnin', 20, 'thin', 1, 'seed', 4);
assert(~isfield(r, 'nowcast') && isequal(size(r.b_mean), [T 5]), 'tvp_midas: the output without a row to nowcast');
assert(max(max(abs(r.b_mean - squeeze(mean(r.draws.b, 1))))) < 1e-12 && ...
    max(max(abs(r.theta_mean{2} - squeeze(mean(r.draws.theta{2}, 1))))) < 1e-12, ...
    'tvp_midas: the posterior means are not the means of the draws');

% no information about the weights: the variances of their random walks keep the prior
% IG(20, 19*.001), whose mean is .001
W0 = {zeros(n,5), zeros(n,5)};
r = ssm.models.tvp_midas(y, X, W0, A, 'tvb', false, 'sv', false, 'nsim', 3000, 'burnin', 1000, ...
    'seed', 6, 'prior', struct('xi2', [20 19*.001]));
assert(all(abs(r.xi2_mean(:)/.001 - 1) < .15), ...
    'tvp_midas: without information the variances of the weights leave their prior');

% the forecast law: four periods with rows that change and the period to nowcast, the
% variances held near .001; the target's weights have the moments of the random walk
% conditioned on all five restrictions at once, which differ from those of a step from
% the fourth period's weights projected onto the target's restriction
T4 = 4; n4 = T4 + 1; Vth = 1; xi = .001;
A4 = [1 + rand(n4,1), randn(n4,4)];
y4 = randn(T4,1); X4 = ones(n4,1);
r = ssm.models.tvp_midas(y4, X4, {zeros(n4,5)}, {A4}, 'tvb', false, 'sv', false, 'nsim', 4000, ...
    'burnin', 2000, 'seed', 7, 'prior', struct('Vtheta', Vth, 'xi2', [1e5 (1e5-1)*xi]));
L = min((1:n4)', 1:n4) - 1;
S = kron(Vth + xi*L, eye(5));                         % the random walk over the five periods
M = kron(eye(n4), ones(1,5)).*repmat(reshape(A4', 1, []), n4, 1);
G = S*M'/(M*S*M');
mu = G*ones(n4,1); Sc = S - G*M*S;
i5 = 5*T4 + (1:5);
mj = mu(i5); Sj = Sc(i5,i5);                          % all five restrictions at once
G4 = S(1:5*T4,1:5*T4)*M(1:T4,1:5*T4)'/(M(1:T4,1:5*T4)*S(1:5*T4,1:5*T4)*M(1:T4,1:5*T4)');
S4 = S(1:5*T4,1:5*T4) - G4*M(1:T4,1:5*T4)*S(1:5*T4,1:5*T4);
a = A4(n4,:)'; D = xi*eye(5); P = eye(5) - D*(a*a')/(a'*D*a);
Sl = P*(S4(end-4:end,end-4:end) + D)*P';              % the step and the projection
th = r.nowcast.theta{1};
assert(norm(cov(th) - Sj, 'fro') < .1*norm(Sj, 'fro') && norm(mean(th)' - mj) < .1*norm(mj), ...
    'tvp_midas: the target''s weights are not those of the random walk conditioned on every restriction');
assert(norm(Sl - Sj, 'fro') > .5*norm(Sj, 'fro'), 'test_tvp_midas: the forecast law check has no power');

% a restriction on the level alone fixes the first weight, whose variance keeps its prior
Ae = repmat({[ones(n,1), zeros(n,4)]}, 1, 2);
r = ssm.models.tvp_midas(y, X, W, Ae, 'tvb', false, 'sv', false, 'nsim', 3000, 'burnin', 1000, ...
    'seed', 8, 'prior', struct('xi2', [20 19*.001]));
assert(all(abs(r.xi2_mean(:,1)/.001 - 1) < .15), ...
    'tvp_midas: the variance of a fixed weight leaves its prior');

% the default prior, given explicitly, changes nothing; a changed prior changes the draws
r0 = ssm.models.tvp_midas(y, X, W, A, 'nsim', 20, 'burnin', 10, 'seed', 5);
r1 = ssm.models.tvp_midas(y, X, W, A, 'nsim', 20, 'burnin', 10, 'seed', 5, 'prior', struct('Vb', 10));
r2 = ssm.models.tvp_midas(y, X, W, A, 'nsim', 20, 'burnin', 10, 'seed', 5, 'prior', struct('Vb', 10 + 1e-9));
assert(isequal(r0.nowcast.m, r1.nowcast.m) && ~isequal(r0.nowcast.m, r2.nowcast.m), ...
    'tvp_midas: the prior option does not act as documented');

% malformed input
Abad = {A{1}, [-A{2}(:,1), A{2}(:,2:end)]};
bad = {{y, [X; X(1,:)], W, A}, {y, X, W, Abad}, {y, X, W, A, 'nsims', 10}, ...
    {y, X, W, A, 'prior', struct('V', 1)}, {y, X, W, A, 'thin', 1.5}, ...
    {y, X, W, A, 'sv', [true false]}, {y, X, W, A, 'quantiles', [0 .5]}, ...
    {y, X, W, A, 'prior', struct('xi2', [10 -1])}, {y, X, W, A, 'prior', struct('Vb', [1 2])}, ...
    {y, X, W, A, 'prior', struct('xi2', repmat([10 .001], 3, 1))}, {ones(T,1), X, W, A}};
ids = [{'ssm:models:tvp_midas:badData', 'ssm:models:tvp_midas:badData'}, ...
    repmat({'ssm:models:tvp_midas:badOption'}, 1, 8), {'ssm:models:tvp_midas:badData'}];
for i = 1:numel(bad)
    try
        ssm.models.tvp_midas(bad{i}{:});
        error('tvp_midas: malformed input %d passed', i);
    catch err
        assert(strcmp(err.identifier, ids{i}), err.message);
    end
end
end

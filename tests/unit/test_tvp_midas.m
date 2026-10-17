function test_tvp_midas
% ssm.models.tvp_midas must reproduce, draw for draw, the sampler with which the MIDAS
% nowcasting tutorial's results were computed, held verbatim in tests/unit/private/
% (midas_sampler.m and xi2_mh.m), in each of the tutorial's five configurations; keep
% the restrictions to rounding error; average its kept draws into its posterior means;
% respond to a change in a prior; and reject malformed input.

rng(3);
T = 40; n = T + 1;                                   % T quarters and one to nowcast
d = (1:30*(n+3))';                                    % daily dates, in days
wend = 30*(4:n+3)'; wstart = wend - 90;               % three-month windows
W = cell(1,2); A = W;
for j = 1:2
    [W{j}, A{j}] = ssm.midas_basis(d, randn(numel(d),1), wstart, wend);
end
X = [ones(n,1), randn(n,2)];
y = X(1:T,:)*[.5; .3; -.2] + .8*W{1}(1:T,:)*[1; .2; 0; 0; 0] ...
    - .5*W{2}(1:T,:)*[1; 0; .1; 0; 0] + .5*randn(T,1);
Wo = cellfun(@(w) w(1:T,:), W, 'UniformOutput', false);
Ao = cellfun(@(a) a(1:T,:), A, 'UniformOutput', false);
xT = struct('x0', X(n,:), 'w', {{W{1}(n,:), W{2}(n,:)}}, 'a', {{A{1}(n,:), A{2}(n,:)}});
nsim = 60; burnin = 140;                              % the step sizes are tuned at sweep 100

% the five models of the tutorial: [tvb tvtheta sv]
cfg = [0 0 0; 0 0 1; 1 1 1; 1 0 1; 0 1 1];
for c = 1:size(cfg, 1)
    opts = struct('tvb', cfg(c,1) == 1, 'tvtheta', cfg(c,2) == 1, 'sv', cfg(c,3) == 1, 'thin', 3);
    rng(10 + c, 'twister');
    f = midas_sampler(y, X(1:T,:), Wo, Ao, xT, opts, nsim, burnin);
    r = ssm.models.tvp_midas(y, X, W, A, 'tvb', opts.tvb, 'tvtheta', opts.tvtheta, ...
        'sv', opts.sv, 'nsim', nsim, 'burnin', burnin, 'thin', 3, 'seed', 10 + c);
    assert(isequal(r.nowcast.m, f.m) && isequal(r.nowcast.v, f.v), ...
        'tvp_midas: the nowcast draws differ from the tutorial sampler, model %d', c);
    assert(isequal(r.draws.b, f.B) && isequal(r.draws.theta, f.Th), ...
        'tvp_midas: the path draws differ from the tutorial sampler, model %d', c);
    assert(isequal(r.max_resid, f.res) && isequal(r.acc, f.acc), ...
        'tvp_midas: the residual or the acceptance rates differ from the tutorial sampler, model %d', c);
    assert(r.max_resid < 1e-10, 'tvp_midas: the restrictions are not met, model %d', c);
end

% the posterior means average the kept draws; no row to nowcast, no nowcast
r = ssm.models.tvp_midas(y, X(1:T,:), Wo, Ao, 'nsim', 30, 'burnin', 20, 'thin', 1, 'seed', 4);
assert(~isfield(r, 'nowcast') && isequal(size(r.b_mean), [T 5]), 'tvp_midas: the output without a row to nowcast');
assert(max(max(abs(r.b_mean - squeeze(mean(r.draws.b, 1))))) < 1e-12 && ...
    max(max(abs(r.theta_mean{2} - squeeze(mean(r.draws.theta{2}, 1))))) < 1e-12, ...
    'tvp_midas: the posterior means are not the means of the draws');

% the default prior, given explicitly, changes nothing; a changed prior changes the draws
r0 = ssm.models.tvp_midas(y, X, W, A, 'nsim', 20, 'burnin', 10, 'seed', 5);
r1 = ssm.models.tvp_midas(y, X, W, A, 'nsim', 20, 'burnin', 10, 'seed', 5, 'prior', struct('Vb', 10));
r2 = ssm.models.tvp_midas(y, X, W, A, 'nsim', 20, 'burnin', 10, 'seed', 5, 'prior', struct('Vb', 10 + 1e-9));
assert(isequal(r0.nowcast.m, r1.nowcast.m) && ~isequal(r0.nowcast.m, r2.nowcast.m), ...
    'tvp_midas: the prior option does not act as documented');

% malformed input
bad = {{y, [X; X(1,:)], W, A}, {y, X, W, {A{1}, A{2}/2}}, {y, X, W, A, 'nsims', 10}, ...
    {y, X, W, A, 'prior', struct('V', 1)}};
ids = {'ssm:models:tvp_midas:badData', 'ssm:models:tvp_midas:badData', ...
    'ssm:models:tvp_midas:badOption', 'ssm:models:tvp_midas:badOption'};
for i = 1:numel(bad)
    try
        ssm.models.tvp_midas(bad{i}{:});
        error('tvp_midas: malformed input %d passed', i);
    catch err
        assert(strcmp(err.identifier, ids{i}), err.message);
    end
end
end

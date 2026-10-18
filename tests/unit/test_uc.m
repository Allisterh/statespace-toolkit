function test_uc
% ssm.models.uc and ssm.models.uc_forecast must reproduce, draw for draw, the sampler and
% the forecasts with which the UC tutorial's results were computed, held verbatim in
% tests/unit/private/ (uc_sampler.m, uc_model.m and uc_forecast.m), in each of the
% tutorial's four models; give the same average as y_{T+1} one period ahead; match, at every
% horizon, forecasts simulated forward from the same draws; and reject malformed input.

rng(5);
T = 80;
y = 2 + cumsum(.3*randn(T,1)) + randn(T,1);
names = {'UC', 'UC-SVgap', 'UCSV', 'UC-MA'};
opts = {{'gap', 'const', 'trend', 'const'}, {'gap', 'rw', 'trend', 'const'}, ...
    {'gap', 'rw', 'trend', 'rw'}, {'gap', 'ar1', 'trend', 'const', 'ma', true}};
nsim = 150; burnin = 100; thin = 10;
for k = 1:4
    m = uc_model(names{k});
    rng(20 + k, 'twister');
    out = uc_sampler(y, m, nsim, burnin, thin);
    f = uc_forecast(m, out);
    r = ssm.models.uc(y, opts{k}{:}, 'nsim', nsim, 'burnin', burnin, 'thin', thin, ...
        'horizon', 4, 'quantiles', [], 'seed', 20 + k);
    assert(isequal(r.params, out.phi) && isequal(r.last, out.last) && isequal(r.acc, out.acc), ...
        'uc: the parameter draws differ from the tutorial sampler, %s', names{k});
    assert(isequal(r.draws.tau, out.tau) && isequal(r.draws.sdh, out.sdh) && isequal(r.draws.sdg, out.sdg), ...
        'uc: the path draws differ from the tutorial sampler, %s', names{k});
    assert(isequal(r.param_names, m.free), 'uc: the parameter names differ, %s', names{k});
    assert(isequal(r.forecast.m(:,1), f.m1) && isequal(r.forecast.v(:,1), f.v1) && ...
        isequal(r.forecast.mavg, f.m4) && isequal(r.forecast.vavg, f.v4), ...
        'uc: the forecasts differ from the tutorial''s, %s', names{k});
end

% the default prior, given explicitly, changes nothing; a changed prior changes the draws
ra = ssm.models.uc(y, opts{1}{:}, 'nsim', 20, 'burnin', 10, 'seed', 6, 'horizon', 0);
rb = ssm.models.uc(y, opts{1}{:}, 'nsim', 20, 'burnin', 10, 'seed', 6, 'horizon', 0, ...
    'prior', struct('gap', struct('S', 2)));
rc = ssm.models.uc(y, opts{1}{:}, 'nsim', 20, 'burnin', 10, 'seed', 6, 'horizon', 0, ...
    'prior', struct('gap', struct('S', 2 + 1e-9)));
assert(isequal(ra.params, rb.params) && ~isequal(ra.params, rc.params), ...
    'uc: the prior option does not act as documented');

% one period ahead, the average is y_{T+1}
r1 = ssm.models.uc(y, 'nsim', 50, 'burnin', 20, 'seed', 1, 'horizon', 1, 'quantiles', []);
assert(isequal(r1.forecast.mavg, r1.forecast.m(:,1)) && isequal(r1.forecast.vavg, r1.forecast.v(:,1)), ...
    'uc: the one-period average differs from y_{T+1}');

% every horizon against paths simulated forward from the same draws (UC-MA and UCSV)
p = (.1:.1:.9)'; R = 40; H = 4;
for k = [3 4]
    r = ssm.models.uc(y, opts{k}{:}, 'nsim', 500, 'burnin', 100, 'seed', 30 + k, 'horizon', 0);
    mm = cell(R, 1); vv = mm; sims = zeros(500*R, H);
    for i = 1:R
        fr = ssm.models.uc_forecast(r, H); mm{i} = fr.m; vv{i} = fr.v;
        sims((i-1)*500 + (1:500), :) = simulate(r, H);
    end
    mm = vertcat(mm{:}); vv = vertcat(vv{:});
    for h = 1:H
        q = quantile(sims(:,h), p);
        Fq = arrayfun(@(x) mean(normcdf((x - mm(:,h))./sqrt(vv(:,h)))), q);
        assert(max(abs(Fq - p)) < .02, 'uc_forecast: y_{T+%d} differs from simulated paths, %s', h, names{k});
    end
end

% malformed input
bad = {{y, 'gap', 'garch'}, {[y; NaN]}, {y, 'prior', struct('tau1', struct('a1', 0))}, {y, 'draws', 1}};
ids = {'ssm:models:uc:badOption', 'ssm:models:uc:badData', 'ssm:models:uc:badOption', ...
    'ssm:models:uc:badOption'};
for i = 1:numel(bad)
    try
        ssm.models.uc(bad{i}{:});
        error('uc: malformed input %d passed', i);
    catch err
        assert(strcmp(err.identifier, ids{i}), err.message);
    end
end
end

function ys = simulate(r, H)
% one path of y_{T+1}, ..., y_{T+H} per draw, simulating every state forward
n = size(r.params, 1); ph = r.params; j = 0;
x = cell(1, 2); xT = {r.last(:,2), r.last(:,3)}; c = {'gap', 'trend'};
for i = 1:2
    k = r.spec.(c{i}); x{i} = zeros(n, H); prev = xT{i};
    for q = 1:H
        switch k.type
            case 'const', prev = ph(:,j+1);
            case 'rw',    prev = prev + sqrt(exp(ph(:,j+2))).*randn(n,1);
            case 'ar1',   prev = ph(:,j+1) + tanh(ph(:,j+2)).*(prev - ph(:,j+1)) + sqrt(exp(ph(:,j+3))).*randn(n,1);
        end
        x{i}(:,q) = prev;
    end
    j = j + struct('const', 1, 'rw', 2, 'ar1', 3).(k.type);
end
if r.spec.ma, psi = tanh(ph(:,end)); else, psi = zeros(n,1); end
tau = r.last(:,1); u = r.last(:,4); ys = zeros(n, H);
for q = 1:H
    tau = tau + sqrt(exp(x{2}(:,q))).*randn(n,1);
    un = sqrt(exp(x{1}(:,q))).*randn(n,1);
    ys(:,q) = tau + un + psi.*u; u = un;
end
end

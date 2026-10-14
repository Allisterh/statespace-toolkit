%% build - regenerate the numbers of tutorials/midas_nowcast/README.md
%
% Nowcasts of US real GDP growth for 1990Q1-2019Q4, made at the end of each month of the
% quarter (h = 2/3, 1/3 and 0), from an AR(2) and five MIDAS regressions of Chan, Poon
% and Zhu (2026): MIDAS, MIDAS-SV, TVP-MIDAS-SV, and TVP-MIDAS-SV-R1 and -R2, in which
% only the coefficients or only the weights vary over time. The predictors are monthly
% industrial production, the weekly NFCI and the daily 10-year minus 3-month spread.
% Every model is re-estimated at every origin on today's data with the release lags of
% the paper's Table 2 (pseudo real time). Part 0 checks the data. Part 1 checks
% TVP-MIDAS-SV on data generated from it, with every variance estimated. Part 2 runs the
% nowcasts, in parallel when the Parallel Computing Toolbox is available, each with its
% own seed and its own file in runs/, so that a serial run gives the same numbers and an
% interrupted build resumes. Part 3 scores them, over the whole sample and by subperiod,
% writes nowcasts.csv and draws the figures. Everything printed goes to a log in tempdir.
%
% Usage, from anywhere:  run tutorials/midas_nowcast/build.m

tdir = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(tdir));
logf = fullfile(tempdir, 'ssm_midas_nowcast_build_log.txt');
if exist(logf, 'file'), delete(logf); end
diary(logf);
fprintf('build.m of tutorials/midas_nowcast, %s, MATLAB %s\n', char(datetime('now')), version);
t0 = tic;
run(fullfile(repo, 'setup.m'));

% settings
seed = 20261004;
nsim = 10000; burnin = 5000;                    % the paper's chain length
nworkers = 8;                                   % parallel workers, when the toolbox exists
cver = 2;                                       % in every cache key
targets = 1990:.25:2019.75;                     % 1990Q1-2019Q4
models = struct( ...
    'name', {'AR(2)', 'MIDAS', 'MIDAS-SV', 'TVP-MIDAS-SV', 'TVP-MIDAS-SV-R1', 'TVP-MIDAS-SV-R2'}, ...
    'file', {'AR2', 'MIDAS', 'MIDAS_SV', 'TVP_MIDAS_SV', 'TVP_MIDAS_SV_R1', 'TVP_MIDAS_SV_R2'}, ...
    'tvb', {false, false, false, true, true, false}, ...
    'tvtheta', {false, false, false, true, false, true}, ...
    'sv', {false, false, true, true, true, true});
hl = {'2/3', '1/3', '0'};
qlab = @(q) sprintf('%dQ%d', floor(q), round(4*(q - floor(q))) + 1);

%% Part 0: the data
fprintf('\n=== Part 0: the data ===\n');
R = cell(1,3);
for j = 1:3
    R{j} = nowcast_rows(repo, j);
    r = R{j};
    assert(~any(isnan([r.y; r.y2; r.X0(:); r.W{1}(:); r.W{2}(:); r.A{1}(:); r.A{2}(:)])), ...
        'build: missing values in the rows of h = %s', hl{j});
    assert(all(abs([r.A{1}(:,1); r.A{2}(:,1)] - 1) < 1e-12), 'build: the first basis must average to 1');
    fprintf(['h = %-3s %d quarters, %s to %s; weekly observations per window %d-%d, ' ...
        'daily %d-%d\n'], hl{j}, numel(r.y), qlab(r.q(1)), qlab(r.q(end)), min(r.m(:,1)), ...
        max(r.m(:,1)), min(r.m(:,2)), max(r.m(:,2)));
end
% Table 2 of the paper: for 2000Q1 the latest industrial production is December, January
% and February at the three origins, and every quarter shifts the same way
i0 = find(R{1}.q == 2000);
assert(isequal([R{1}.iplast(i0) R{2}.iplast(i0) R{3}.iplast(i0)], ...
    datetime(1999,12,1) + calmonths(0:2)), 'build: the information sets differ from Table 2');
assert(isequal(R{2}.X0(:,3:4), R{1}.X0(:,4:5)) && isequal(R{3}.X0(:,3:4), R{2}.X0(:,4:5)), ...
    'build: the monthly regressors do not move one month per origin');
assert(all(R{1}.y == R{2}.y & R{2}.y == R{3}.y), 'build: the horizons must share the target');
% with a_jt of full column rank, theta = e_1 (equal weights) is the only fixed theta that
% meets every quarter's restriction a_jt'*theta = 1
for j = 1:3
    assert(rank(R{j}.A{1}) == 5 && rank(R{j}.A{2}) == 5, 'build: the basis averages lack full rank');
end
fprintf('the basis averages of each predictor have rank 5: of the fixed weights, only equal ones meet every quarter''s restriction\n');
fprintf('information sets of Table 2: industrial production to %s, %s and %s for 2000Q1\n', ...
    char(R{1}.iplast(i0), 'MMM yyyy'), char(R{2}.iplast(i0), 'MMM yyyy'), ...
    char(R{3}.iplast(i0), 'MMM yyyy'));

%% Part 1: TVP-MIDAS-SV on generated data, with the regressors of h = 0 to 2019Q3
fprintf('\n=== Part 1: TVP-MIDAS-SV on generated data ===\n');
rng(seed, 'twister');
r = R{3}; n = find(abs(r.q - 2019.75) < 1e-9) - 1;
X0 = r.X0(1:n,:); W = {r.W{1}(1:n,:), r.W{2}(1:n,:)}; A = {r.A{1}(1:n,:), r.A{2}(1:n,:)};
om2 = 1e-3*ones(7,1); xi2 = 1e-4*ones(5,1); sig2g = .01;   % near their prior means
Btrue = [1 .3 .2 .2 .2 -1 .5] + [zeros(1,7); cumsum(sqrt(om2').*randn(n-1,7))];
% the weights from their prior, a random walk with theta_1 ~ N(0, 10 I) conditioned on
% every restriction a_t'*theta_t = 1, as in the model
H5 = kron(ssm.diffmat(n), speye(5));
P5 = H5'*spdiags([ones(5,1)/10; repmat(1./xi2, n-1, 1)], 0, 5*n, 5*n)*H5;
Thtrue = cell(1,2);
for k = 1:2
    [th, ~, C5] = ssm.simulate_states(P5, zeros(5*n,1));
    M5 = sparse(repelem((1:n)', 5), (1:5*n)', reshape(A{k}', [], 1), n, 5*n);
    Thtrue{k} = reshape(ssm.restrict(th, C5, M5, ones(n,1)), 5, n)';
end
gtrue = log(2) + [0; cumsum(sqrt(sig2g)*randn(n-1,1))];
yg = sum(X0.*Btrue(:,1:5), 2) + Btrue(:,6).*sum(Thtrue{1}.*W{1}, 2) ...
    + Btrue(:,7).*sum(Thtrue{2}.*W{2}, 2) + exp(gtrue/2).*randn(n,1);
xT = struct('x0', r.X0(n+1,:), 'w', {{r.W{1}(n+1,:), r.W{2}(n+1,:)}}, ...
    'a', {{r.A{1}(n+1,:), r.A{2}(n+1,:)}});
t1 = tic;
fg = midas_sampler(yg, X0, W, A, xT, struct('tvb', true, 'tvtheta', true, 'sv', true, ...
    'thin', 10), nsim, burnin);
fprintf('%d quarters, %d draws after %d, %.1f s; largest restriction residual %.1e\n', n, ...
    nsim, burnin, toc(t1), fg.res);
fprintf('%-26s %9s %9s %9s\n', 'paths', 'RMSE', 'post sd', 'coverage');
cvg = zeros(1,3);
cvg(1) = report('coefficients b_t', fg.B, Btrue);
cvg(2) = report('NFCI weights theta_t', fg.Th{1}, Thtrue{1});
cvg(3) = report('spread weights theta_t', fg.Th{2}, Thtrue{2});
assert(all(cvg > .75) && fg.res < 1e-8, 'build: the sampler does not recover the generated paths');

%% Part 2: the nowcasts
fprintf('\n=== Part 2: nowcasts ===\n');
rdir = fullfile(tdir, 'runs');
if ~exist(rdir, 'dir'), mkdir(rdir); end
nm = numel(models); nt = numel(targets);
jobs = zeros(0, 3);                             % [model horizon target]
for k = 1:nm
    for it = 1:nt
        for j = 1:3
            if k == 1 && j > 1, continue; end   % the AR(2) nowcast is the same at every h
            jobs(end+1,:) = [k j it]; %#ok<SAGROW>
        end
    end
end
heavy = ismember(jobs(:,1), find([models.tvtheta]));
[~, ord] = sortrows([heavy jobs(:,3)], 'descend');   % the longest runs first
jobs = jobs(ord,:);
nj = size(jobs, 1);
nw = 0;
if license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel'))
    c = parcluster('Processes');
    c.NumWorkers = max(c.NumWorkers, nworkers);
    p = gcp('nocreate');
    if isempty(p), p = parpool(c, nworkers); end
    nw = p.NumWorkers;
end
fprintf('%d runs on %d parallel workers (0: serial), %d draws after %d each\n', nj, nw, ...
    nsim, burnin);
t2 = tic;
out = zeros(nj, 8);
parfor (ii = 1:nj, nw)
    out(ii,:) = one_job(tdir, R, models, jobs(ii,:), targets, nsim, burnin, seed, cver);
end
fprintf('done in %.1f minutes; the runs themselves took %.1f hours in all\n', toc(t2)/60, ...
    sum(out(:,8))/3600);
assert(max(out(:,7)) < 1e-8, 'build: a restriction is not met');
fprintf('largest restriction residual over every draw of every nowcast: %.1e\n', max(out(:,7)));

%% Part 3: scores
fprintf('\n=== Part 3: scores ===\n');
S = nan(nm, nt, 3, 6);                          % [point lpl q05 q10 q50 q95]
for ii = 1:nj
    k = jobs(ii,1); j = jobs(ii,2); it = jobs(ii,3);
    if k == 1, S(k,it,:,:) = repmat(reshape(out(ii,1:6), 1, 1, 1, 6), 1, 1, 3);
    else, S(k,it,j,:) = out(ii,1:6); end
end
iq = arrayfun(@(q) find(abs(R{1}.q - q) < 1e-9), targets)';
yv = R{1}.y(iq);
fprintf(['%d nowcasts per model and horizon, %s to %s; relative to the AR(2): RMSFE ratio, ' ...
    'difference in the sum of log predictive likelihoods, and ratio of the mean quantile ' ...
    'score at 10%%\n'], nt, qlab(targets(1)), qlab(targets(end)));
for j = 1:3
    fprintf('\nh = %s\n| Model | RMSFE | relative RMSFE | log predictive likelihood | difference | quantile score | relative QS | 90%% coverage |\n|---|---|---|---|---|---|---|---|\n', hl{j});
    rm = zeros(nm,1); lp = rm; qs = rm;
    for k = 1:nm
        rm(k) = sqrt(mean((squeeze(S(k,:,j,1))' - yv).^2));
        lp(k) = sum(S(k,:,j,2));
        q10 = squeeze(S(k,:,j,4))';
        qs(k) = mean(((yv <= q10) - .1).*(q10 - yv));
        cv = mean(yv >= squeeze(S(k,:,j,3))' & yv <= squeeze(S(k,:,j,6))');
        fprintf('| %s | %.3f | %.3f | %.1f | %.1f | %.3f | %.3f | %.2f |\n', models(k).name, ...
            rm(k), rm(k)/rm(1), lp(k), lp(k) - lp(1), qs(k), qs(k)/qs(1), cv);
    end
end

% quarter by quarter, and by subperiod
tq = targets';
lpd = S(:,:,:,2);                               % model x target x horizon
pe = S(:,:,:,1) - yv';                          % point errors
fprintf('\nshare of the quarters in which each model beats the AR(2) in log predictive density, h = 2/3, 1/3, 0\n');
for k = 2:nm
    fprintf('%-16s %s\n', models(k).name, sprintf('%.2f ', mean(lpd(k,:,:) > lpd(1,:,:), 2)));
end
sp = {tq < 2008, tq >= 2008}; spl = {'1990Q1-2007Q4', '2008Q1-2019Q4'};
fprintf('\nrelative RMSFE against the AR(2) by subperiod\n| Model | %s, h = 2/3 | 1/3 | 0 | %s, h = 2/3 | 1/3 | 0 |\n|---|---|---|---|---|---|---|\n', spl{:});
for k = 2:nm
    v = zeros(1,6);
    for p = 1:2
        for j = 1:3
            v(3*(p-1)+j) = sqrt(mean(pe(k,sp{p},j).^2))/sqrt(mean(pe(1,sp{p},j).^2));
        end
    end
    fprintf('| %s | %s\n', models(k).name, sprintf('%.3f | ', v));
end
% pairs of models that differ in one feature: the sum of the differences in log
% predictive density, by subperiod and over the whole sample
pairs = [5 3; 4 5; 6 3];
pname = {'coefficients vary: TVP-MIDAS-SV-R1 minus MIDAS-SV', ...
    'weights vary: TVP-MIDAS-SV minus TVP-MIDAS-SV-R1', ...
    'weights vary: TVP-MIDAS-SV-R2 minus MIDAS-SV'};
fprintf('\ndifference in the sum of log predictive likelihoods\n| Comparison | %s, h = 2/3 | 1/3 | 0 | %s, h = 2/3 | 1/3 | 0 | all, h = 2/3 | 1/3 | 0 |\n|---|---|---|---|---|---|---|---|---|---|\n', spl{:});
for p = 1:3
    dl = lpd(pairs(p,1),:,:) - lpd(pairs(p,2),:,:);
    v = [squeeze(sum(dl(1,sp{1},:), 2))', squeeze(sum(dl(1,sp{2},:), 2))', squeeze(sum(dl, 2))'];
    fprintf('| %s | %s\n', pname{p}, sprintf('%.1f | ', v));
end

% every nowcast, for reuse: nowcasts.csv next to this file
[kk, tt, jj] = ndgrid(1:nm, 1:nt, 1:3);
P = reshape(permute(S, [1 2 3 4]), [], 6);
fcsv = table(reshape({models(kk(:)).name}, [], 1), arrayfun(qlab, targets(tt(:))', ...
    'UniformOutput', false), hl(jj(:))', yv(tt(:)), P(:,1), P(:,3), P(:,4), P(:,5), P(:,6), ...
    P(:,2), 'VariableNames', {'model', 'target', 'horizon', 'realized', 'point_nowcast', ...
    'q05', 'q10', 'q50', 'q95', 'log_predictive_density'});
writetable(fcsv, fullfile(tdir, 'nowcasts.csv'));
fprintf('\nnowcasts.csv: %d rows written\n', height(fcsv));

% the figures
cols = lines(nm);
lname = {'MIDAS', 'MIDAS-SV', 'TVP-MIDAS-SV (coefficients and weights vary)', ...
    'TVP-MIDAS-SV-R1 (coefficients vary, fixed weights)', ...
    'TVP-MIDAS-SV-R2 (fixed coefficients, weights vary)'};
fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 18 10]); hold on
yline(0, 'Color', [.6 .6 .6]);
hc = gobjects(1, nm-1);
for k = 2:nm
    hc(k-1) = plot(tq, cumsum(squeeze(S(k,:,3,2)) - squeeze(S(1,:,3,2))), 'Color', cols(k,:), ...
        'LineWidth', 1.3);
end
hold off; box off; xlim(tq([1 end]));
ylabel('cumulative log score gain over the AR(2)');
legend(hc, lname, 'Location', 'northwest'); legend boxoff
exportgraphics(fig, fullfile(tdir, 'fig_score.png'), 'Resolution', 150); close(fig);

fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 18 9]);
hcol = lines(3);
ax = gobjects(1,2);
ptitle = {{'coefficients vary:', 'TVP-MIDAS-SV-R1 minus MIDAS-SV'}, ...
    {'weights vary:', 'TVP-MIDAS-SV minus TVP-MIDAS-SV-R1'}};
for p = 1:2
    ax(p) = subplot(1, 2, p); hold on
    yline(0, 'Color', [.6 .6 .6]);
    hh = gobjects(1,3);
    for j = 1:3
        hh(j) = plot(tq, cumsum(squeeze(lpd(pairs(p,1),:,j)) - squeeze(lpd(pairs(p,2),:,j))), ...
            'Color', hcol(j,:), 'LineWidth', 1.3);
    end
    hold off; box off; xlim(tq([1 end])); title(ptitle{p}, 'FontWeight', 'normal');
    if p == 1
        ylabel('cumulative difference in log score');
        legend(hh, strcat({'h = '}, hl), 'Location', 'northwest'); legend boxoff
    end
end
linkaxes(ax, 'y');
exportgraphics(fig, fullfile(tdir, 'fig_compare.png'), 'Resolution', 150); close(fig);

fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 18 9]); hold on
hb = ssm.shaded_band(tq, squeeze(S(5,:,3,3)), squeeze(S(5,:,3,6)));
hn = plot(tq, squeeze(S(5,:,3,1)), 'Color', cols(5,:), 'LineWidth', 1.3);
ha = plot(tq, yv, 'k.', 'MarkerSize', 10);
hold off; box off; xlim(tq([1 end]));
ylabel('annualized GDP growth (%)');
legend([ha hn hb], {'GDP growth', 'nowcast, TVP-MIDAS-SV-R1 at h = 0', ...
    '90% predictive interval'}, 'Location', 'southwest'); legend boxoff
exportgraphics(fig, fullfile(tdir, 'fig_nowcast.png'), 'Resolution', 150); close(fig);

fprintf('\nbuild finished in %.1f minutes\n', toc(t0)/60);
diary off

%% local functions
function r = one_job(tdir, R, models, job, targets, nsim, burnin, seed, cver)
% one model, horizon and target: load the result saved under its key, or compute and save
% it. r = [point lpl q05 q10 q50 q95 residual seconds]
k = job(1); j = job(2); it = job(3);
rr = R{j}; iq = find(abs(rr.q - targets(it)) < 1e-9);
sd = seed + 100000*k + 1000*j + iq;
file = fullfile(tdir, 'runs', sprintf('%s_h%d_%d.mat', models(k).file, j, iq));
key = [nsim burnin sd cver];
if exist(file, 'file')
    s = load(file);
    if isequal(s.key, key), r = s.r; return; end
end
% one computational thread, as on a parallel worker, so serial and parallel runs agree
nt = maxNumCompThreads(1);
restore = onCleanup(@() maxNumCompThreads(nt));
rng(sd, 'twister');
t1 = tic;
o = 1:iq-1;
if k == 1
    f = ar_nowcast(rr.y(o), [ones(numel(o),1) rr.X0(o,2) rr.y2(o)], ...
        [1 rr.X0(iq,2) rr.y2(iq)], nsim, burnin);
    f.res = 0;
else
    xT = struct('x0', rr.X0(iq,:), 'w', {{rr.W{1}(iq,:), rr.W{2}(iq,:)}}, ...
        'a', {{rr.A{1}(iq,:), rr.A{2}(iq,:)}});
    f = midas_sampler(rr.y(o), rr.X0(o,:), {rr.W{1}(o,:), rr.W{2}(o,:)}, ...
        {rr.A{1}(o,:), rr.A{2}(o,:)}, xT, models(k), nsim, burnin);
end
y = rr.y(iq);
l = -.5*log(2*pi*f.v) - .5*(y - f.m).^2./f.v;
q = arrayfun(@(p) mixquantile(p, f.m, f.v), [.05 .1 .5 .95]);
r = [mean(f.m), max(l) + log(mean(exp(l - max(l)))), q, f.res, toc(t1)];
save(file, 'key', 'r');
end

function x = mixquantile(p, m, v)
% the p-quantile of the equally weighted mixture of N(m_i, v_i)
s = sqrt(v);
x = fzero(@(q) mean(normcdf((q - m)./s)) - p, [min(m - 8*s) max(m + 8*s)]);
end

function c = report(name, D, truth)
% RMSE of the posterior mean, mean posterior sd and 90% coverage of draws D (draws x ...)
% against truth
m = squeeze(mean(D, 1)); sdv = squeeze(std(D, 0, 1));
b = quantile(D, [.05 .95], 1);
c = mean(truth(:) >= reshape(b(1,:,:), [], 1) & truth(:) <= reshape(b(2,:,:), [], 1));
fprintf('%-26s %9.3f %9.3f %8.1f%%\n', name, sqrt(mean((m(:) - truth(:)).^2)), mean(sdv(:)), 100*c);
end

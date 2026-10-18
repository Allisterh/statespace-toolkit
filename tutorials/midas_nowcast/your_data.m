%% your_data - nowcast a quarterly series with MIDAS regressions, on your own data
%
% Runs the comparison of build.m, an AR(2) and the MIDAS regressions of Chan, Poon and
% Zhu (2026) listed in `models`, on a quarterly target with monthly predictors, each
% entering as its three latest available months, and weekly or daily predictors, each
% with a Fourier weighting function over the three months ending at the origin. The last
% nfc quarters are nowcast at the end of each month of the quarter (h = 2/3, 1/3 and 0)
% and scored by the root mean squared forecast error and the sum of log predictive
% likelihoods. With latest = true each model also nowcasts the quarter after the last
% one observed, at the latest month-end the predictors reach: the monthly ones up to their
% release lag, and each weekly or daily one with a release in the last seven days of the
% month. The defaults use the tutorial's data with short chains and take about a minute;
% the numbers on the page come from build.m. The report goes to outdir, tempdir by
% default.
%
% Every file is a csv with a date column. The target is in levels turned into 400 times
% their log changes (qtransform = 'dlog400') or used as it is ('none'). Keep the target
% in percent, as such log changes are, since the volatility step adds 1e-4 to each
% squared error before it takes the log. The monthly predictors are columns of one file,
% log-differenced (mtransform = 'dlog') or not ('none') and then multiplied by mscale; a
% numeric date column is read as year + month/12, the convention of the FRED-MD file.
% The weekly and daily predictors are used as they are, each value from its release date:
% hrelease holds, for each of them, a function that turns its dates into release dates,
% or [] when its values are released on their dates. Each predictor must cover the
% quarters from qfirst to the end of the sample, the monthly ones from five months
% earlier.

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
run(fullfile(repo, 'setup.m'));
lastwarn('');

%% ---- settings ----
d = fullfile(repo, 'examples', 'data');
qfile = fullfile(d, 'USGDP.csv'); qcol = 'GDPC1'; qtransform = 'dlog400';
mfile = fullfile(d, 'FRED-MD.csv'); mcols = {'INDPRO'}; mtransform = 'none'; mscale = 100;
mlag = 1;                          % months by which the monthly predictors are released late
hfiles = {fullfile(d, 'NFCI_weekly.csv'), fullfile(d, 'T10Y3M_daily.csv')};
hcols = {'NFCI', 'T10Y3M'};
hrelease = {@nfci_release, []};    % the release dates of each weekly or daily predictor
qfirst = datetime(1982,4,1);       % the first quarter of the estimation sample
models = {'AR(2)'                  % delete a row to leave a model out
          'MIDAS'
          'MIDAS-SV'
          'TVP-MIDAS-SV'
          'TVP-MIDAS-SV-R1'
          'TVP-MIDAS-SV-R2'};
nfc = 2;                           % the last nfc observed quarters are nowcast; 20 must remain
nsim = 200; burnin = 100;          % draws per model, quarter and horizon
seed = 1;
latest = true;                     % also nowcast the quarter after the last one observed
outdir = tempdir;                  % where the report goes; '' for none

%% ---- data ----
[qd, v] = read_series(qfile, qcol);
qd = dateshift(qd, 'start', 'quarter');
switch qtransform
    case 'dlog400'
        assert(all(v(~isnan(v)) > 0), 'your_data: qtransform = ''dlog400'' needs positive levels');
        y = 400*diff(log(v)); qd = qd(2:end);
    case 'none', y = v;
    otherwise, error('your_data: qtransform must be ''dlog400'' or ''none''');
end
keep = ~isnan(y); y = y(find(keep, 1):find(keep, 1, 'last')); qd = qd(find(keep, 1):find(keep, 1, 'last'));
assert(~any(isnan(y)) && all(calmonths(between(qd(1:end-1), qd(2:end))) == 3), ...
    'your_data: the target must be quarterly, without gaps');
assert(var(y) > 0, 'your_data: the target is constant');
[md, X] = read_series(mfile, mcols);
md = dateshift(md, 'start', 'month');
switch mtransform
    case 'dlog'
        assert(all(X(~isnan(X)) > 0), 'your_data: mtransform = ''dlog'' needs positive levels');
        X = diff(log(X)); md = md(2:end);
    case 'none'
    otherwise, error('your_data: mtransform must be ''dlog'' or ''none''');
end
X = mscale*X;
ok = all(~isnan(X), 2); md = md(ok); X = X(ok,:);
assert(all(calmonths(between(md(1:end-1), md(2:end))) == 1), ...
    'your_data: the monthly predictors must be monthly, without gaps');
assert(numel(hcols) == numel(hfiles) && numel(hrelease) == numel(hfiles), ...
    'your_data: hfiles, hcols and hrelease must have one entry per weekly or daily predictor');
hf = cell(1, numel(hfiles));
for k = 1:numel(hfiles)
    [hd, hx] = read_series(hfiles{k}, hcols{k});
    hf{k} = struct('d', hd(~isnan(hx)), 'x', hx(~isnan(hx)));
    if ~isempty(hrelease{k}), hf{k}.r = hrelease{k}(hf{k}.d); end
end
first = find(qd == qfirst);
assert(isscalar(first) && first > 2, 'your_data: qfirst must be a quarter of the target, after its second');
R = cell(1,3);
for j = 1:3, R{j} = nowcast_design(qd, y, struct('d', md, 'X', X), hf, j, mlag, first); end
assert(all(R{1}.m(:) > 0) && all(R{3}.m(:) > 0), ...
    'your_data: a weekly or daily predictor has no observation in some window');
T = numel(R{1}.y);
assert(isscalar(nfc) && nfc >= 1 && nfc == fix(nfc) && T - nfc >= 20, ...
    'your_data: nfc must be a positive integer that leaves at least 20 quarters before the first nowcast');
qlab = @(q) sprintf('%dQ%d', floor(q), round(4*(q - floor(q))) + 1);
fprintf('\n%s, %s to %s, %d quarters; %d monthly and %d weekly or daily predictors\n', qcol, ...
    qlab(R{1}.q(1)), qlab(R{1}.q(end)), T, numel(mcols), numel(hf));

%% ---- nowcasts of the last nfc quarters ----
flags = containers.Map({'MIDAS', 'MIDAS-SV', 'TVP-MIDAS-SV', 'TVP-MIDAS-SV-R1', 'TVP-MIDAS-SV-R2'}, ...
    {[0 0 0], [0 0 1], [1 1 1], [1 0 1], [0 1 1]});
names = {'AR(2)', 'MIDAS', 'MIDAS-SV', 'TVP-MIDAS-SV', 'TVP-MIDAS-SV-R1', 'TVP-MIDAS-SV-R2'};
assert(all(ismember(models, names)), 'your_data: models may hold only %s', strjoin(names, ', '));
nm = numel(models);
tg = T - nfc + 1 : T;
res = zeros(nm, 3, 2);                         % [RMSFE LPL] by model and horizon
for k = 1:nm
    id = find(strcmp(models{k}, names));       % the seeds follow the model, whatever its row
    for j = 1:3
        if id == 1 && j > 1                    % the AR(2) nowcast is the same at every h
            res(k,j,:) = res(k,1,:);
            continue
        end
        e = zeros(numel(tg),1); l = e;
        for i = 1:numel(tg)
            rng(seed + 1000*id + 100*j + tg(i), 'twister');
            f = one_model(models{k}, flags, R{j}, tg(i), nsim, burnin);
            e(i) = mean(f.m) - R{j}.y(tg(i));
            lp = -.5*log(2*pi*f.v) - .5*(R{j}.y(tg(i)) - f.m).^2./f.v;
            l(i) = max(lp) + log(mean(exp(lp - max(lp))));
        end
        res(k,j,:) = [sqrt(mean(e.^2)) sum(l)];
    end
end
fprintf('\n%d nowcasts per model and horizon, %s to %s\n', nfc, qlab(R{1}.q(tg(1))), qlab(R{1}.q(T)));
fprintf('| Model | RMSFE, h = 2/3, 1/3, 0 | log predictive likelihood, h = 2/3, 1/3, 0 |\n|---|---|---|\n');
for k = 1:nm
    fprintf('| %s | %.2f, %.2f, %.2f | %.1f, %.1f, %.1f |\n', models{k}, res(k,:,1), res(k,:,2));
end

%% ---- the nowcast of the next quarter ----
fut = nan(nm, 3); jl = NaN; ql = ''; %#ok<NASGU>
if latest
    qn = qd(end) + calmonths(3);
    jl = latest_origin(qn, md, mlag, hf);      % the latest month-end the predictors reach
    if isnan(jl)
        fprintf('\nthe predictors do not reach into %s; no nowcast of it\n', qlab(year(qn) + (quarter(qn)-1)/4));
    else
        Rn = nowcast_design([qd; qn], [y; NaN], struct('d', md, 'X', X), hf, jl, mlag, first);
        ql = qlab(Rn.q(end));
        hl = {'2/3', '1/3', '0'};
        fprintf('\nnowcasts of %s at the end of its month %d (h = %s)\n', ql, jl, hl{jl});
        fprintf('latest values used: the monthly predictors %s', char(Rn.iplast(end), 'MMM yyyy'));
        for k = 1:numel(hf)
            fprintf(', %s %s', hcols{k}, char(Rn.hlast(end,k), 'd MMM yyyy'));
        end
        fprintf('\n| Model | nowcast | 90%% interval |\n|---|---|---|\n');
        for k = 1:nm
            rng(seed + 1000*find(strcmp(models{k}, names)) + 999, 'twister');
            f = one_model(models{k}, flags, Rn, numel(Rn.y), nsim, burnin);
            fut(k,:) = [mean(f.m), mixquantile(.05, f.m, f.v), mixquantile(.95, f.m, f.v)];
            fprintf('| %s | %.2f | (%.2f, %.2f) |\n', models{k}, fut(k,:));
        end
    end
end

%% ---- the report ----
if ~isempty(outdir)
    if ~isfolder(outdir), mkdir(outdir); end
    report = array2table([reshape(res, nm, []) fut], 'VariableNames', {'rmsfe_h2_3', ...
        'rmsfe_h1_3', 'rmsfe_h0', 'lpl_h2_3', 'lpl_h1_3', 'lpl_h0', 'nowcast', ...
        'nowcast_lower_90', 'nowcast_upper_90'});
    report = [table(models, 'VariableNames', {'model'}) report];
    [msg, id] = lastwarn;
    meta = struct('qfile', qfile, 'qcol', qcol, 'qtransform', qtransform, 'mfile', mfile, ...
        'mcols', {mcols}, 'mtransform', mtransform, 'mscale', mscale, 'mlag', mlag, ...
        'hfiles', {hfiles}, 'hcols', {hcols}, 'hrelease', {cellfun(@release_rule, hrelease, ...
        'UniformOutput', false)}, 'qfirst', qfirst, 'nfc', nfc, 'nsim', nsim, ...
        'burnin', burnin, 'seed', seed, 'nowcast_quarter', ql, 'nowcast_month', jl, ...
        'run_time', datetime('now'), 'matlab_version', version, 'last_warning', msg, ...
        'last_warning_id', id);
    writetable(report, fullfile(outdir, 'midas_nowcast_report.csv'));
    save(fullfile(outdir, 'midas_nowcast_report.mat'), 'report', 'meta');
    fprintf('\nreport written to %s and .mat\n', fullfile(outdir, 'midas_nowcast_report.csv'));
end

%% ---- local functions ----
function f = one_model(name, flags, r, iq, nsim, burnin)
% one model estimated on the quarters before row iq of r, nowcasting row iq
o = 1:iq-1;
if strcmp(name, 'AR(2)')
    f = ar_nowcast(r.y(o), [ones(numel(o),1) r.X0(o,2) r.y2(o)], [1 r.X0(iq,2) r.y2(iq)], nsim, burnin);
else
    fl = flags(name);
    J = numel(r.W);                            % row iq of X, W and A is the quarter to nowcast
    fr = ssm.models.tvp_midas(r.y(o), r.X0(1:iq,:), cellfun(@(w) w(1:iq,:), r.W, 'UniformOutput', false), ...
        cellfun(@(a) a(1:iq,:), r.A, 'UniformOutput', false), 'tvb', fl(1) == 1, ...
        'tvtheta', fl(2) == 1 && J > 0, 'sv', fl(3) == 1, 'nsim', nsim, 'burnin', burnin);
    f = struct('m', fr.nowcast.m, 'v', fr.nowcast.v);
end
end

function [dt, v] = read_series(file, cols)
% the date column (the first) and the named columns of a csv file; a numeric date column
% is read as year + month/12, and text dates as yyyy-MM-dd or M/d/yyyy
o = detectImportOptions(file, 'VariableNamingRule', 'preserve');
o = setvartype(o, o.VariableNames{1}, 'char');
t = readtable(file, o);
dc = string(t{:,1});
num = str2double(dc);
if all(~isnan(num))
    dt = datetime(floor(num), round(12*(num - floor(num))) + 1, 1) - calmonths(1);
elseif all(contains(dc, '/'))
    dt = datetime(dc, 'InputFormat', 'M/d/yyyy');
else
    dt = datetime(dc, 'InputFormat', 'yyyy-MM-dd');
end
v = t{:, cellstr(cols)};
end

function x = mixquantile(p, m, v)
% the p-quantile of the equally weighted mixture of N(m_i, v_i)
s = sqrt(v);
x = fzero(@(q) mean(normcdf((q - m)./s)) - p, [min(m - 8*s) max(m + 8*s)]);
end

function s = release_rule(f)
% the release rule of a weekly or daily predictor, as text for the report
if isempty(f), s = 'on its dates'; else, s = func2str(f); end
end

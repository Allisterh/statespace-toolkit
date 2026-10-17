function R = nowcast_design(qd, y, mon, hf, j, mlag, first)
% nowcast_design - the regressors of every quarter for nowcasts made at the end of month j
% (1, 2 or 3) of the quarter, with the information set of Table 2 of Chan, Poon and Zhu
% (2026) applied to every quarter alike. Quarter t has the target y_t and the regressors
% [1, y_{t-1}, the three latest months of each monthly predictor available at the origin],
% and, for each high-frequency predictor, W{k}(t,:) and A{k}(t,:) from ssm.midas_basis
% over the three months ending at the origin.
%   qd, y    : Q x 1 first days of the quarters and the target (y(1:2) only as lags)
%   mon      : monthly predictors, fields d (first days of the months) and X (months x km)
%   hf       : 1 x J cell of high-frequency predictors, each with fields d (dates) and x
%   mlag     : the months by which the monthly predictors are released late
%   first    : the index in qd of the first quarter to return; it needs y(first-2)
%   R.q, R.y, R.y2 : T x 1 quarter labels (year + (quarter-1)/4), y_t and y_{t-2}
%   R.X0     : T x (2 + 3*km);  R.W, R.A : 1 x J cells of T x 5;  R.m : T x J counts
%   R.iplast : T x 1 month of the latest monthly value used
qs = (first:numel(qd))';
T = numel(qs); J = numel(hf); km = size(mon.X, 2);
R.q = year(qd(qs)) + (quarter(qd(qs)) - 1)/4;
R.y = y(qs);
R.y2 = y(qs-2);
R.X0 = zeros(T, 2 + 3*km); R.W = repmat({zeros(T,5)}, 1, J); R.A = R.W; R.m = zeros(T, J);
R.iplast = NaT(T, 1);
oend = NaT(T, 1); wstart = NaT(T, 1);
for i = 1:T
    qstart = qd(qs(i));
    oend(i) = qstart + calmonths(j) - caldays(1);           % the origin: end of month j
    wstart(i) = qstart + calmonths(j-3) - caldays(1);       % the window is (wstart, oend]
    mo = find(mon.d == dateshift(oend(i), 'start', 'month')); % the origin's month
    assert(isscalar(mo) && mo - mlag - 2 >= 1, 'nowcast_design: no monthly data for %s', ...
        char(oend(i)));
    R.X0(i,:) = [1, y(qs(i)-1), reshape(mon.X(mo-mlag-2:mo-mlag, :), 1, [])];
    R.iplast(i) = mon.d(mo-mlag);
end
for k = 1:J
    [R.W{k}, R.A{k}, R.m(:,k)] = ssm.midas_basis(hf{k}.d, hf{k}.x, wstart, oend);
end
end

function R = nowcast_design(qd, y, mon, hf, j, mlag, first)
% nowcast_design - the regressors of every quarter for nowcasts made at the end of month j
% (1, 2 or 3) of the quarter, with the information set of Table 2 of Chan, Poon and Zhu
% (2026) applied to every quarter alike. Quarter t has the target y_t and the regressors
% [1, y_{t-1}, the three latest months of each monthly predictor available at the origin],
% and, for each high-frequency predictor, the observations of the three months ending at
% the origin. An observation dated d sits at s = (origin - d)/(window length), where the
% Fourier basis v(s) = [1, cos(2*pi*s), sin(2*pi*s), cos(4*pi*s), sin(4*pi*s)]' is
% evaluated; W{k}(t,:) is the mean over the window of v(s)*x and A{k}(t,:) the mean of
% v(s), so that the restriction A{k}(t,:)*theta = 1 makes the weights theta'*v(s)/m_t
% sum to 1.
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
for i = 1:T
    qstart = qd(qs(i));
    oend = qstart + calmonths(j) - caldays(1);              % the origin: end of month j
    wstart = qstart + calmonths(j-3) - caldays(1);          % the window is (wstart, oend]
    mo = find(mon.d == dateshift(oend, 'start', 'month'));  % the origin's month
    assert(isscalar(mo) && mo - mlag - 2 >= 1, 'nowcast_design: no monthly data for %s', ...
        char(oend));
    R.X0(i,:) = [1, y(qs(i)-1), reshape(mon.X(mo-mlag-2:mo-mlag, :), 1, [])];
    R.iplast(i) = mon.d(mo-mlag);
    L = days(oend - wstart);
    for k = 1:J
        in = hf{k}.d > wstart & hf{k}.d <= oend;
        s = days(oend - hf{k}.d(in))/L;
        V = [ones(size(s)), cos(2*pi*s), sin(2*pi*s), cos(4*pi*s), sin(4*pi*s)];
        R.W{k}(i,:) = mean(V.*hf{k}.x(in), 1);
        R.A{k}(i,:) = mean(V, 1);
        R.m(i,k) = nnz(in);
    end
end
end

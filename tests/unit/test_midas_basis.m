function test_midas_basis
% ssm.midas_basis must equal, bit for bit, the window loop of the MIDAS nowcasting
% tutorial's nowcast_design that it replaced, for daily and weekly datetimes at each origin
% of the quarter, and give the same rows for numeric dates; with 'scale', 'sum' it must sum
% where that loop averages, with a first column of A equal to the counts; W = c*A for a
% constant predictor; release dates must keep out the observations released after a
% window ends; NaN values are left out; and an empty window or a bad option stops it.

rng(2);
dd = (datetime(2000,1,1):caldays(1):datetime(2004,12,31))';
dd = dd(rand(numel(dd),1) > .3);                             % daily, with gaps
dw = (datetime(2000,1,7):calweeks(1):datetime(2004,12,31))'; % weekly
qs = (datetime(2000,4,1):calmonths(3):datetime(2004,7,1))';  % quarters
T = numel(qs);
for j = 1:3
    wend = qs + calmonths(j) - caldays(1);                   % the origin: end of month j
    wstart = qs + calmonths(j-3) - caldays(1);
    for k = 1:2
        if k == 1, d = dd; else, d = dw; end
        x = randn(numel(d), 1);
        % the loop of nowcast_design
        W0 = zeros(T,5); A0 = W0; m0 = zeros(T,1);
        for i = 1:T
            L = days(wend(i) - wstart(i));
            in = d > wstart(i) & d <= wend(i);
            s = days(wend(i) - d(in))/L;
            V = [ones(size(s)), cos(2*pi*s), sin(2*pi*s), cos(4*pi*s), sin(4*pi*s)];
            W0(i,:) = mean(V.*x(in), 1);
            A0(i,:) = mean(V, 1);
            m0(i) = nnz(in);
        end
        [W, A, m] = ssm.midas_basis(d, x, wstart, wend);
        [Wm, Am] = ssm.midas_basis(d, x, wstart, wend, 'scale', 'mean');
        assert(isequal(W, W0) && isequal(A, A0) && isequal(m, m0) && isequal(Wm, W) && isequal(Am, A), ...
            'midas_basis: differs from the loop of nowcast_design');
        [Wn, An, mn] = ssm.midas_basis(datenum(d), x, datenum(wstart), datenum(wend));
        assert(max(abs(Wn(:) - W(:))) < 1e-12 && max(abs(An(:) - A(:))) < 1e-12 && isequal(mn, m), ...
            'midas_basis: numeric dates give other rows');
        assert(all(A(:,1) == 1), 'midas_basis: the first column of A is not 1');
        [Ws, As, ms] = ssm.midas_basis(d, x, wstart, wend, 'scale', 'sum');
        assert(isequal(ms, m) && isequal(As(:,1), m) && max(abs(Ws(:) - W(:).*repmat(m, 5, 1))) < 1e-10 ...
            && max(abs(As(:) - A(:).*repmat(m, 5, 1))) < 1e-10, 'midas_basis: ''sum'' does not sum');
        Wc = ssm.midas_basis(d, 2.5*ones(size(x)), wstart, wend);
        assert(max(abs(Wc(:) - 2.5*A(:))) < 1e-12, 'midas_basis: a constant predictor does not give W = c*A');
        % released five days after its date, an observation of the last five days of a
        % window is left out of it
        [Wr, Ar, mr] = ssm.midas_basis(d, x, wstart, wend, 'release', d + caldays(5), 'scale', 'sum');
        [Wl, Al, ml] = deal(zeros(T,5), zeros(T,5), zeros(T,1));
        for i = 1:T
            in = d > wstart(i) & d <= wend(i) - caldays(5);
            s = days(wend(i) - d(in))/days(wend(i) - wstart(i));
            V = [ones(size(s)), cos(2*pi*s), sin(2*pi*s), cos(4*pi*s), sin(4*pi*s)];
            Wl(i,:) = sum(V.*x(in), 1); Al(i,:) = sum(V, 1); ml(i) = nnz(in);
        end
        assert(isequal(Wr, Wl) && isequal(Ar, Al) && isequal(mr, ml) && all(mr <= m) && any(mr < m), ...
            'midas_basis: the release dates do not act as documented');
    end
end

% NaN values are left out
d = dw; x = randn(numel(d), 1); x(5:7) = NaN;
[W1, A1, m1] = ssm.midas_basis(d, x, wstart, wend);
[W2, A2, m2] = ssm.midas_basis(d(~isnan(x)), x(~isnan(x)), wstart, wend);
assert(isequal(W1, W2) && isequal(A1, A2) && isequal(m1, m2), 'midas_basis: NaN values are not left out');

% an empty window, and bad options
bad = {{d, x, datetime(1990,1,1), datetime(1990,3,31)}, {d, x, wstart, wend, 'scale', 'median'}, ...
    {d, x, wstart, wend, 'release'}, {d, x, wstart, wend, 'lag', 5}, ...
    {d, x, wstart, wend, 'release', d(1:end-1)}};
ids = {'ssm:midas_basis:emptyWindow', 'ssm:midas_basis:badOption', 'ssm:midas_basis:badOption', ...
    'ssm:midas_basis:badOption', 'ssm:midas_basis:badData'};
for i = 1:numel(bad)
    try
        ssm.midas_basis(bad{i}{:});
        error('midas_basis: bad input %d passed', i);
    catch err
        assert(strcmp(err.identifier, ids{i}), err.message);
    end
end
end

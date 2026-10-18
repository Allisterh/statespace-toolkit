% ssm.midas_basis - the regressors and restriction rows of one high-frequency predictor in
% the MIDAS regressions of Chan, Poon and Zhu (2026), with the Fourier weighting function,
% over the window of each low-frequency period. Written for this toolkit.
%
%   [W, A, m] = ssm.midas_basis(d, x, wstart, wend)
%   [W, A, m] = ssm.midas_basis(d, x, wstart, wend, 'scale', 'mean', 'release', r)
%
%   d, x   : N x 1 dates and values of the predictor; NaN values are left out
%   wstart, wend : T x 1 ends of the windows; the window of period t is
%            (wstart(t), wend(t)]
%   'scale'   : 'mean' (the default) or 'sum'; see below
%   'release' : N x 1 release dates of the observations, default d; an observation enters
%               the window of period t only if it is released by wend(t)
%   W      : T x 5 matrix whose row t is the mean of v(s)*x over the window of period t,
%            the regressors w_t' of ssm.models.tvp_midas; with 'sum', the sum
%   A      : T x 5 matrix whose row t is the mean of v(s) over that window, the
%            restriction rows a_t'; with 'sum', the sum
%   m      : T x 1 numbers of observations in the windows
%
% An observation dated d sits at s = (wend - d)/(wend - wstart), 0 at the end of its
% window and 1 at the start, and v(s) = [1, cos(2*pi*s), sin(2*pi*s), cos(4*pi*s),
% sin(4*pi*s)]' is the Fourier basis. The weights theta'*v(s)/m(t) of period t then give
% the weighted average theta'*W(t,:)' and sum to A(t,:)*theta. With 'sum', the scaling of
% Chan, Poon and Zhu (2026), the weights are theta'*v(s), so theta is m(t) times smaller
% for the same weights, and the same prior variances on theta allow far larger weights.
% Dates are datetimes, or numbers measured in days. A window without an observation is an
% error.
%
% See:
% Chan, J.C.C., Poon, A. and Zhu, D. (2026). Time-Varying Parameter MIDAS Models:
% Application to Nowcasting US Real GDP, Journal of Econometrics, forthcoming.

function [W, A, m] = midas_basis(d, x, wstart, wend, varargin)
scale = 'mean'; r = [];
iv = 1;
while iv <= numel(varargin)
    if iv == numel(varargin)
        error('ssm:midas_basis:badOption', 'option ''%s'' has no value', char(string(varargin{iv})));
    end
    switch lower(char(string(varargin{iv})))
        case 'scale',   scale = lower(char(string(varargin{iv+1})));
        case 'release', r = varargin{iv+1};
        otherwise, error('ssm:midas_basis:badOption', 'unknown option ''%s''', char(string(varargin{iv})));
    end
    iv = iv + 2;
end
if ~ismember(scale, {'sum', 'mean'})
    error('ssm:midas_basis:badOption', 'scale must be ''sum'' or ''mean''');
end
d = d(:); x = x(:); wstart = wstart(:); wend = wend(:);
if isempty(r), r = d; else, r = r(:); end
if numel(d) ~= numel(x) || numel(r) ~= numel(d)
    error('ssm:midas_basis:badData', 'd, x and the release dates must have the same number of entries');
end
if numel(wstart) ~= numel(wend)
    error('ssm:midas_basis:badData', 'wstart and wend must have the same number of entries');
end
ok = ~isnan(x);
T = numel(wend);
W = zeros(T, 5); A = zeros(T, 5); m = zeros(T, 1);
for t = 1:T
    in = ok & d > wstart(t) & d <= wend(t) & r <= wend(t);
    if ~any(in)
        error('ssm:midas_basis:emptyWindow', 'the window of period %d holds no observation', t);
    end
    L = wend(t) - wstart(t);
    s = wend(t) - d(in);
    if isduration(L), L = days(L); s = days(s); end
    s = s/L;
    V = [ones(size(s)), cos(2*pi*s), sin(2*pi*s), cos(4*pi*s), sin(4*pi*s)];
    if strcmp(scale, 'sum')
        W(t,:) = sum(V.*x(in), 1);
        A(t,:) = sum(V, 1);
    else
        W(t,:) = mean(V.*x(in), 1);
        A(t,:) = mean(V, 1);
    end
    m(t) = nnz(in);
end
end

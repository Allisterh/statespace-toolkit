% ssm.midas_basis - the regressors and restriction rows of one high-frequency predictor in
% the MIDAS regressions of Chan, Poon and Zhu (2026), with the Fourier weighting function,
% over the window of each low-frequency period. Written for this toolkit.
%
%   [W, A, m] = ssm.midas_basis(d, x, wstart, wend)
%
%   d, x   : N x 1 dates and values of the predictor; NaN values are left out
%   wstart, wend : T x 1 ends of the windows; the window of period t is
%            (wstart(t), wend(t)]
%   W      : T x 5 matrix whose row t is the mean of v(s)*x over the window of period t,
%            the regressors w_t' of ssm.models.tvp_midas
%   A      : T x 5 matrix whose row t is the mean of v(s) over that window, the
%            restriction rows a_t'
%   m      : T x 1 numbers of observations in the windows
%
% An observation dated d sits at s = (wend - d)/(wend - wstart), 0 at the end of its
% window and 1 at the start, and v(s) = [1, cos(2*pi*s), sin(2*pi*s), cos(4*pi*s),
% sin(4*pi*s)]' is the Fourier basis. The weights theta'*v(s)/m(t) of period t then give
% the weighted average theta'*W(t,:)' and sum to A(t,:)*theta. Dates are datetimes, or
% numbers measured in days. A window without an observation is an error.
%
% See:
% Chan, J.C.C., Poon, A. and Zhu, D. (2026). Time-Varying Parameter MIDAS Models:
% Application to Nowcasting US Real GDP, Journal of Econometrics, forthcoming.

function [W, A, m] = midas_basis(d, x, wstart, wend)
d = d(:); x = x(:); wstart = wstart(:); wend = wend(:);
if numel(d) ~= numel(x)
    error('ssm:midas_basis:badData', 'd and x must have the same number of entries');
end
if numel(wstart) ~= numel(wend)
    error('ssm:midas_basis:badData', 'wstart and wend must have the same number of entries');
end
ok = ~isnan(x);
T = numel(wend);
W = zeros(T, 5); A = zeros(T, 5); m = zeros(T, 1);
for t = 1:T
    in = ok & d > wstart(t) & d <= wend(t);
    if ~any(in)
        error('ssm:midas_basis:emptyWindow', 'the window of period %d holds no observation', t);
    end
    L = wend(t) - wstart(t);
    s = wend(t) - d(in);
    if isduration(L), L = days(L); s = days(s); end
    s = s/L;
    V = [ones(size(s)), cos(2*pi*s), sin(2*pi*s), cos(4*pi*s), sin(4*pi*s)];
    W(t,:) = mean(V.*x(in), 1);
    A(t,:) = mean(V, 1);
    m(t) = nnz(in);
end
end

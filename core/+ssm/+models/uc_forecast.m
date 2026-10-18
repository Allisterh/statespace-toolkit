% ssm.models.uc_forecast - the predictive distributions of y_{T+1}, ..., y_{T+H} and of their
% average at each posterior draw of ssm.models.uc. Written for this toolkit.
%
%   f = ssm.models.uc_forecast(res, H)
%
%   res : the output of ssm.models.uc, or any struct with its fields spec (spec.gap.type,
%         spec.trend.type and spec.ma), params and last
%   H   : the number of periods ahead
%   f.m, f.v       : nsim x H means and variances of the normal densities of y_{T+h} given
%                    each draw
%   f.mavg, f.vavg : nsim x 1 means and variances of the normal densities of the average of
%                    y_{T+1}, ..., y_{T+H} given each draw
%
% Each draw simulates the log variances H periods ahead from its parameters and from h_T and
% g_T. Given those, y_{T+h} and the average are normal, with the trend and the gap integrated
% out. With G_j = exp(g_{T+j}) and H_j = exp(h_{T+j}), y_{T+h} has mean tau_T + psi*u_T for
% h = 1 and tau_T after, and variance G_1 + ... + G_h + H_h, plus psi^2*H_{h-1} for h > 1; the
% average has mean tau_T + psi*u_T/H and variance
% (sum_j (H-j+1)^2*G_j + (1+psi)^2*(H_1 + ... + H_{H-1}) + H_H)/H^2. The predictive density
% of each target is the equally weighted mixture of these normals over the draws. Random
% numbers: randn(nsim,1) per period ahead, the gap first, for each component whose log
% variance is not constant.

function f = uc_forecast(res, H)
n = size(res.params, 1);
last = res.last; m = res.spec;
[xg, j] = path(m.gap, res.params, 0, last(:,2), n, H);
xt = path(m.trend, res.params, j, last(:,3), n, H);
Hv = exp(xg); G = exp(xt);
if m.ma, psi = tanh(res.params(:,end)); else, psi = zeros(n,1); end
tauT = last(:,1); uT = last(:,4);
f.m = repmat(tauT, 1, H);
f.m(:,1) = tauT + psi.*uT;
f.v = cumsum(G, 2) + Hv;
f.v(:,2:end) = f.v(:,2:end) + psi.^2.*Hv(:,1:end-1);
tv = H^2*G(:,1);
for q = 2:H, tv = tv + (H - q + 1)^2*G(:,q); end
f.mavg = tauT + psi.*uT/H;
f.vavg = (tv + (1 + psi).^2.*sum(Hv(:,1:H-1), 2) + Hv(:,H))/H^2;
end

function [x, j] = path(k, phi, j, xT, n, H)
% the log variance in periods T+1, ..., T+H from the columns of phi after j
x = zeros(n, H);
switch k.type
    case 'const'
        x = repmat(phi(:,j+1), 1, H); j = j + 1;
    case 'rw'
        s = sqrt(exp(phi(:,j+2))); j = j + 2; prev = xT;
        for q = 1:H, prev = prev + s.*randn(n,1); x(:,q) = prev; end
    case 'ar1'
        mu = phi(:,j+1); rho = tanh(phi(:,j+2)); s = sqrt(exp(phi(:,j+3))); j = j + 3;
        prev = xT;
        for q = 1:H, prev = mu + rho.*(prev - mu) + s.*randn(n,1); x(:,q) = prev; end
end
end

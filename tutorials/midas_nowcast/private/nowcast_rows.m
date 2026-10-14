function R = nowcast_rows(repo, j)
% nowcast_rows - the rows of nowcast_design for the tutorial's data, nowcasts made at the
% end of month j of the quarter: GDP growth (400 times the log change of GDPC1, from
% examples/data/USGDP.csv), with industrial production (100 times the log change, from
% the FRED-MD file, released a month late) as the monthly predictor, and the weekly NFCI
% and the daily 10-year minus 3-month spread as the high-frequency predictors. The rows
% start in 1982Q2.
d = fullfile(repo, 'examples', 'data');
o = detectImportOptions(fullfile(d, 'USGDP.csv'));
o = setvartype(o, 1, 'char');
gdp = readtable(fullfile(d, 'USGDP.csv'), o);
gd = datetime(gdp{:,1}, 'InputFormat', 'M/d/yyyy');
yq = 400*diff(log(gdp.GDPC1)); gd = gd(2:end);              % growth, from 1947Q2
md = readtable(fullfile(d, 'FRED-MD.csv'), 'VariableNamingRule', 'preserve');
ip = 100*md.INDPRO;                                         % FRED-MD holds dlog(INDPRO)
ipd = datetime(floor(md.Date), round(12*(md.Date - floor(md.Date))) + 1, 1) - calmonths(1);
assert(all(ismember(days(diff(ipd)), 28:31)), 'nowcast_rows: the FRED-MD dates are not monthly');
nf = readtable(fullfile(d, 'NFCI_weekly.csv'));
sp = readtable(fullfile(d, 'T10Y3M_daily.csv'));
sp = sp(~isnan(sp.T10Y3M), :);
hf = {struct('d', nf.observation_date, 'x', nf.NFCI), struct('d', sp.observation_date, 'x', sp.T10Y3M)};
R = nowcast_design(gd, yq, struct('d', ipd, 'X', ip), hf, j, 1, find(gd == datetime(1982,4,1)));
end

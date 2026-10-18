function test_midas_nowcast_helpers
% The helpers of the MIDAS nowcasting tutorial (tutorials/midas_nowcast/private), run from
% a copy: nowcast_design reads the monthly predictors at a ragged edge and leaves out the
% weekly values not yet released at the origin; latest_origin rejects a month that a
% predictor does not complete; and nfci_release gives the release days of the NFCI,
% Thursday in a week with a federal holiday.

root = getappdata(0, 'ssm_repo_root');
if isempty(root), root = fileparts(fileparts(fileparts(mfilename('fullpath')))); end
tmp = tempname; mkdir(tmp);
for f = {'nowcast_design.m', 'latest_origin.m', 'nfci_release.m'}
    copyfile(fullfile(root, 'tutorials', 'midas_nowcast', 'private', f{1}), tmp);
end
addpath(tmp);
cleanup = onCleanup(@() remove(tmp));

% nfci_release: Wednesday, or Thursday after Memorial Day, a Sunday Independence Day
% observed on Monday, Columbus Day, New Year's Day, Martin Luther King Jr. Day and
% Christmas Day; Thanksgiving falls on a Thursday and moves nothing
fri = datetime([2017 6 2; 2017 5 26; 2021 7 2; 2026 10 9; 2019 12 27; 2000 1 14; 2018 12 21; ...
    2018 11 16]);
rel = datetime([2017 6 7; 2017 6 1; 2021 7 8; 2026 10 15; 2020 1 2; 2000 1 20; 2018 12 27; ...
    2018 11 21]);
assert(isequal(nfci_release(fri), rel), 'nfci_release: wrong release days');
try
    nfci_release(datetime(2017, 6, 1));
    error('nfci_release: a Thursday passed');
catch err
    assert(contains(err.message, 'Fridays'), err.message);
end

% nowcast_design at a ragged edge: the monthly data end in December 2025 and the origin
% is 31 January 2026, so the rows use October to December
qd = (datetime(2000,1,1):calmonths(3):datetime(2026,1,1))';
y = randn(numel(qd), 1);
md = (datetime(1999,1,1):calmonths(1):datetime(2025,12,1))';
X = (1:numel(md))';                                 % the value of a month is its index
dd = (datetime(1999,1,1):caldays(1):datetime(2026,1,31))';
wk = (datetime(1999,1,1):calweeks(1):datetime(2026,1,31))';   % Fridays
hf = {struct('d', dd, 'x', randn(numel(dd), 1)), ...
    struct('d', wk, 'x', randn(numel(wk), 1), 'r', wk + caldays(5))};
R = nowcast_design(qd, y, struct('d', md, 'X', X), hf, 1, 1, 3);
assert(R.iplast(end) == datetime(2025,12,1) && isequal(R.X0(end,3:5), ...
    find(ismember(md, datetime(2025,10:12,1)))'), 'nowcast_design: wrong months at the ragged edge');

% the weekly values enter from their release: at the origin 31 January 2026, a Saturday,
% the value of Friday 30 January is not out, and that of 23 January is
assert(R.hlast(end,2) == datetime(2026,1,23) && R.hlast(end,1) == datetime(2026,1,31), ...
    'nowcast_design: a weekly value is used before its release');
Rd = nowcast_design(qd, y, struct('d', md, 'X', X), {hf{1}, rmfield(hf{2}, 'r')}, 1, 1, 3);
assert(Rd.hlast(end,2) == datetime(2026,1,30) && all(Rd.m(:,2) - R.m(:,2) >= 0), ...
    'nowcast_design: without release dates the weekly values enter on their dates');
% the scales of ssm.midas_basis
Rs = nowcast_design(qd, y, struct('d', md, 'X', X), hf, 1, 1, 3, 'sum');
assert(all(R.A{1}(:,1) == 1) && isequal(Rs.A{1}(:,1), Rs.m(:,1)), 'nowcast_design: wrong scale');

% latest_origin: March is complete only when every predictor reaches its last seven days
qn = datetime(2025,1,1); mdl = (datetime(2020,1,1):calmonths(1):datetime(2025,11,1))';
dly = @(last) struct('d', (datetime(2020,1,1):caldays(1):last)');
wkl = @(last) struct('d', (datetime(2020,1,3):calweeks(1):last)');
assert(latest_origin(qn, mdl, 1, {dly(datetime(2025,3,3)), wkl(datetime(2025,3,7))}) == 2, ...
    'latest_origin: an incomplete March passed');
assert(latest_origin(qn, mdl, 1, {dly(datetime(2025,3,28)), wkl(datetime(2025,3,28))}) == 3, ...
    'latest_origin: a complete March failed');
assert(latest_origin(qn, mdl, 1, {dly(datetime(2025,3,31)), wkl(datetime(2025,3,21))}) == 2, ...
    'latest_origin: a weekly series without the last week of March passed');
assert(isnan(latest_origin(qn, mdl(mdl < datetime(2024,12,1)), 1, {dly(datetime(2025,3,31))})), ...
    'latest_origin: a quarter without monthly data passed');
end

function remove(tmp)
% take the copies off the path and delete them
rmpath(tmp);
rmdir(tmp, 's');
end

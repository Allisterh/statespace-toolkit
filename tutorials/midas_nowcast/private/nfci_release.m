function r = nfci_release(d)
% nfci_release - the release dates of the weekly values of the NFCI dated d, the Fridays
% that end their weeks. The Federal Reserve Bank of Chicago releases each value on the
% Wednesday after its Friday, or on the Thursday when a federal holiday falls on the Monday,
% Tuesday or Wednesday of that week. A holiday on a Sunday counts on the Monday after.
assert(all(weekday(d) == 6), 'nfci_release: the dates must be Fridays');
r = d + caldays(5);
hol = federal_holidays(unique(year([d; r])));
late = ismember(d + caldays(3), hol) | ismember(d + caldays(4), hol) | ismember(d + caldays(5), hol);
r(late) = r(late) + caldays(1);
end

function h = federal_holidays(yrs)
% the federal holidays of the years yrs that can fall on a Monday, Tuesday or Wednesday, a
% Sunday one moved to the Monday after: New Year's Day, Martin Luther King Jr. Day (from
% 1986), Washington's Birthday, Memorial Day, Juneteenth (from 2021), Independence Day,
% Labor Day, Columbus Day, Veterans Day and Christmas Day
h = NaT(0, 1);
for y = yrs(:)'
    fixed = datetime(y, [1 7 11 12], [1 4 11 25])';
    if y >= 2021, fixed(end+1) = datetime(y, 6, 19); end %#ok<AGROW>
    fixed(weekday(fixed) == 1) = fixed(weekday(fixed) == 1) + caldays(1);
    mondays = [nthmonday(y, 2, 3); lastmonday(y, 5); nthmonday(y, 9, 1); nthmonday(y, 10, 2)];
    if y >= 1986, mondays(end+1) = nthmonday(y, 1, 3); end %#ok<AGROW>
    h = [h; fixed; mondays]; %#ok<AGROW>
end
end

function d = nthmonday(y, m, n)
% the nth Monday of month m of year y
first = datetime(y, m, 1);
d = first + caldays(mod(2 - weekday(first), 7) + 7*(n - 1));
end

function d = lastmonday(y, m)
% the last Monday of month m of year y
last = dateshift(datetime(y, m, 1), 'end', 'month');
d = last - caldays(mod(weekday(last) - 2, 7));
end

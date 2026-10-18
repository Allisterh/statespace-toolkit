function j = latest_origin(qn, md, mlag, hf)
% latest_origin - the latest month j (1, 2 or 3) of the quarter that starts on qn at whose
% end every predictor is complete, or NaN if there is none. The monthly predictors must
% hold the month mlag months before the origin's month. Each weekly or daily predictor must
% have a value released in the last seven days of the month, which always hold a weekly
% release and a business day.
%   md : the first days of the months of the monthly predictors
%   hf : 1 x J cell of the weekly and daily predictors, each with the field d (dates), and
%        r (release dates) when its values are released after their dates
j = NaN;
for m = 3:-1:1
    oend = qn + calmonths(m) - caldays(1);
    if any(md == dateshift(oend, 'start', 'month') - calmonths(mlag)) && ...
            all(cellfun(@(h) max(released(h)) > oend - caldays(7), hf))
        j = m;
        return
    end
end
end

function r = released(h)
% the release dates of the values of a weekly or daily predictor
if isfield(h, 'r'), r = h.r; else, r = h.d; end
end

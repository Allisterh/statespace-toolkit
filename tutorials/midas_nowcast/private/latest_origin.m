function j = latest_origin(qn, md, mlag, hf)
% latest_origin - the latest month j (1, 2 or 3) of the quarter that starts on qn at whose
% end every predictor is complete, or NaN if there is none. The monthly predictors must
% hold the month mlag months before the origin's month. Each weekly or daily predictor
% must reach into the last seven days of the month, which always hold a weekly value and
% a business day.
%   md : the first days of the months of the monthly predictors
%   hf : 1 x J cell of the weekly and daily predictors, each with the field d (dates)
j = NaN;
for m = 3:-1:1
    oend = qn + calmonths(m) - caldays(1);
    if any(md == dateshift(oend, 'start', 'month') - calmonths(mlag)) && ...
            all(cellfun(@(h) max(h.d) > oend - caldays(7), hf))
        j = m;
        return
    end
end
end

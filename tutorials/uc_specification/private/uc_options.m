function o = uc_options(name)
% uc_options - the options of ssm.models.uc for the tutorial's models UC, UC-SVgap, UCSV and
% UC-MA, whose priors are the function's defaults; the other models have samplers of their own
switch name
    case 'UC',       o = {'gap', 'const', 'trend', 'const'};
    case 'UC-SVgap', o = {'gap', 'rw', 'trend', 'const'};
    case 'UCSV',     o = {'gap', 'rw', 'trend', 'rw'};
    case 'UC-MA',    o = {'gap', 'ar1', 'trend', 'const', 'ma', true};
    otherwise, error('uc_options: no model %s', name);
end
end
